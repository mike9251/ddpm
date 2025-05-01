import logging
import os
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from unet import UNet
from dataloader import get_dataloader
from loggers import TensorboardLogger
from schedulers import LinearNoiseScheduler
from meters import RunningMeter
from utils import set_seed


logging.basicConfig(filename=None, encoding="utf-8", level=logging.DEBUG)


class Trainer:
    def __init__(self, config: DictConfig):
        self.output_dir = Path(config["output_dir"])
        self.log_every = config["log_every"]
        self.logger = TensorboardLogger(self.output_dir / "logs")
        self.last_epoch = -1
        self.epochs = config["epochs"]
        self.num_img_to_sample = config["num_img_to_sample"]
        self.grad_accum_steps = config["grad_accum_steps"]

        self.device_id = config.get("device_id", 0)
        self.rank = config.get("rank", 0)
        self.ddp = config["ddp"]
        self.world_size = 1

        self.device = torch.device(config["device"] + f":{self.device_id}")

        self.unet = UNet(config["time_dim"], self.device).to(self.device)

        if self.ddp:
            self.unet = DDP(self.unet, device_ids=[self.device_id], find_unused_parameters=False)
            self.world_size = dist.get_world_size()
        
        self.opt = torch.optim.Adam(self._unwrap().parameters(), lr=config["lr"])

        if config.resume_from is not None:
            self._load_state(config.resume_from)

        self.dataloader = get_dataloader(config["data_dir"], config["img_size"], config["batch_size"], config["num_workers"], config["ddp"])
        self.noise_scheduler = LinearNoiseScheduler(config)

        self.running_meters = {"train/running/mse_loss": RunningMeter(window_size=self.log_every, ddp=self.ddp)}
        self.epoch_meters = {"train/epoch/mse_loss": RunningMeter(window_size=len(self.dataloader) // self.world_size, ddp=self.ddp)}

    def _unwrap(self):
        if self.ddp:
            return self.unet.module
        return self.unet
    
    def _load_state(self, ckpt_path: Path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self._unwrap().load_state_dict(ckpt["unet"])
        self.opt.load_state_dict(ckpt["opt"])
        self.last_epoch = ckpt["epoch"]
        logging.info(f"Resume training from {ckpt_path} from last epoch {self.last_epoch}")

    def _save_checkpoint(self, epoch: int):
        if self.rank != 0:
            return

        os.makedirs(self.output_dir / "checkpoints", exist_ok=True)

        torch.save({"unet": self._unwrap().state_dict(),
                    "opt": self.opt.state_dict(),
                    "epoch": epoch
                    }, self.output_dir / "checkpoints" / f"unet_epoch_{epoch}.pt")
    
    def train(self):
        start_epoch = self.last_epoch + 1
        steps_per_epoch = len(self.dataloader)
        for epoch in range(start_epoch, self.epochs):
            if self.ddp:
                self.dataloader.sampler.set_epoch(epoch)

            self.opt.zero_grad()

            with tqdm(range(steps_per_epoch), disable=not self.rank == 0) as pbar:
                for i, x0 in enumerate(self.dataloader):
                    global_step = i + steps_per_epoch * epoch
                    x0 = x0.to(self.device)
                    t = self.noise_scheduler.sample_timesteps(x0.shape[0])
                    xt, noise = self.noise_scheduler.noise_image(x0, t)

                    noise_pred = self.unet(xt, t)

                    loss = F.mse_loss(noise_pred, noise)

                    loss.backward()

                    if global_step > 0 and global_step % self.grad_accum_steps == 0:
                        self.opt.step()
                        self.opt.zero_grad()

                    self.running_meters["train/running/mse_loss"].update(loss.detach().cpu().item())
                    self.epoch_meters["train/epoch/mse_loss"].update(loss.detach().cpu().item())

                    if (i + 0) % self.log_every == 0:
                        logs = {tag: meter.compute() for tag, meter in self.running_meters.items()}

                        imgs = self.noise_scheduler.sample(self.unet, self.num_img_to_sample)
                        logs["img"] = imgs.detach().cpu()

                        if self.rank == 0:
                            self.logger.log(logs, global_step)
                    
                    pbar.set_postfix(
                        EPOCH=epoch,
                        MSE_LOSS=np.round(
                            self.running_meters["train/running/mse_loss"].compute(), 5
                        )
                    )
                    pbar.update(1)
                    break


            logs = {tag: meter.compute() for tag, meter in self.epoch_meters.items()}
            # imgs = self.noise_scheduler.sample(self.unet, self.num_img_to_sample)
            # logs["img"] = imgs.detach().cpu()
            if self.rank == 0:
                self.logger.log(logs, global_step)

                self._save_checkpoint(epoch)

            self.last_epoch = epoch


@hydra.main(config_path="../configs/", config_name="train_ddpm_uncond.yaml", version_base="1.3")
def main(config: DictConfig):
    set_seed(3910574)

    if config.ddp:
        logging.info("Setting up DDP!")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device_id = rank % torch.cuda.device_count()

        OmegaConf.set_struct(config, True)
        with open_dict(config):
            config.rank = rank
            config.device_id = device_id

    trainer = Trainer(config)

    logging.info("Start training!")
    trainer.train()
    logging.info("Done training!")

    if config.ddp:
        dist.destroy_process_group()
        logging.info("Cleaned up DDP!")


if __name__ == "__main__":
    main()
