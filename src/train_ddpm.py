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
import copy

from ema import EMA
from unet import UNet
from dataloader import get_dataloader
from loggers import TensorboardLogger
from schedulers import LinearNoiseScheduler
from meters import RunningMeter
from utils import set_seed


logging.basicConfig(filename=None, encoding="utf-8", level=logging.DEBUG)


from math import log, floor


def human_format(number):
    units = ['', 'K', 'M', 'G', 'T', 'P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%.2f%s' % (number / k**magnitude, units[magnitude])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Trainer:
    def __init__(self, config: DictConfig):
        self.output_dir = Path(config["output_dir"])
        self.log_every = config["log_every"]
        self.logger = TensorboardLogger(self.output_dir / "logs")
        self.last_epoch = -1
        self.epochs = config["epochs"]
        self.num_img_to_sample = config["num_img_to_sample"]

        self.device_id = config.get("device_id", 0)
        self.rank = config.get("rank", 0)
        self.ddp = config["ddp"]
        self.world_size = 1

        self.device = torch.device(config["device"] + f":{self.device_id}")

        self.unet = UNet(time_dim=config["time_dim"], width=config["width"], num_classes=config["num_classes"], device=self.device).to(self.device)
        self.unet.train()

        self.ema = EMA(beta=config["beta_ema"], step=config["step_ema"], start_step=config["start_step_ema"])
        self.ema_unet = copy.deepcopy(self.unet).eval().requires_grad_(False)

        print(f"Number of parameters: {human_format(count_parameters(self.unet))}")

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
        self._unwrap().load_state_dict(ckpt["unet_ema"])
        self.opt.load_state_dict(ckpt["opt"])
        self.last_epoch = ckpt["epoch"]
        logging.info(f"Resume training from {ckpt_path} from last epoch {self.last_epoch}")

    def _save_checkpoint(self, epoch: int):
        if self.rank != 0:
            return

        os.makedirs(self.output_dir / "checkpoints", exist_ok=True)

        torch.save({"unet_ema": self._unwrap().state_dict(),
                    "opt": self.opt.state_dict(),
                    "epoch": epoch
                    }, self.output_dir / "checkpoints" / f"unet_epoch_{epoch}_{(epoch + 1) * len(self.dataloader)}_ema_{self.ema.start_step}.pt")
    
    def train(self):
        start_epoch = self.last_epoch + 1
        steps_per_epoch = len(self.dataloader)
        for epoch in range(start_epoch, self.epochs):
            if self.ddp:
                self.dataloader.sampler.set_epoch(epoch)

            with tqdm(range(steps_per_epoch), disable=not self.rank == 0) as pbar:
                for i, x0 in enumerate(self.dataloader):
                    global_step = i + steps_per_epoch * epoch
                    x0 = x0.to(self.device)
                    t = self.noise_scheduler.sample_timesteps(x0.shape[0])
                    xt, noise = self.noise_scheduler.noise_image(x0, t)

                    # y = torch.randint(low=0, high=10, size=(x0.shape[0],), device=self.device)

                    noise_pred = self.unet(xt, t)#, y)

                    loss = F.mse_loss(noise_pred, noise)

                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

                    self.ema.update_ema_model(self.unet, self.ema_unet)

                    self.running_meters["train/running/mse_loss"].update(loss.detach().cpu().item())
                    self.epoch_meters["train/epoch/mse_loss"].update(loss.detach().cpu().item())

                    if global_step % self.log_every == 0:
                        logs = {tag: meter.compute() for tag, meter in self.running_meters.items()}

                        sampled_images = self.noise_scheduler.sample_ema(self.unet, self.ema_unet, self.num_img_to_sample)
                        logs.update({title: imgs.detach().cpu() for title, imgs in sampled_images.items()})

                        # ema_imgs = self.noise_scheduler.sample(self.unet, self.num_img_to_sample)
                        # logs["img"] = imgs.detach().cpu()

                        if self.rank == 0:
                            self.logger.log(logs, global_step)
                    
                    pbar.set_postfix(
                        EPOCH=epoch,
                        MSE_LOSS=np.round(
                            self.running_meters["train/running/mse_loss"].compute(), 5
                        )
                    )
                    pbar.update(1)

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
