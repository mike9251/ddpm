import logging
import os
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from ema import EMA
from unet import UNet
from dataloader import get_dataloader
from loggers import TensorboardLogger
from schedulers import LinearNoiseScheduler
from meters import RunningMeter
from utils import set_seed
import torchvision
import cv2
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
        self.ckpt_every = config["ckpt_every"]

        self.device_id = config.get("device_id", 0)
        self.rank = config.get("rank", 0)
        self.ddp = config["ddp"]
        self.world_size = 1
        self.num_classes = config["num_classes"]
        self.cfg_scale = config.get("cfg_scale", 0)
        self.use_weighted_loss = config.get("use_weighted_loss", False)

        self.device = torch.device(config["device"] + f":{self.device_id}")

        self.unet = UNet(time_dim=config["time_dim"], width=config["width"], num_classes=config["num_classes"], device=self.device).to(self.device)
        self.unet.train()

        self.ema = EMA(beta=config["beta_ema"], start_step=config["start_step_ema"])
        self.ema_unet = UNet(time_dim=config["time_dim"], width=config["width"], num_classes=config["num_classes"], device=self.device).to(self.device).eval().requires_grad_(False)

        if self.ddp:
            self.unet = DDP(self.unet, device_ids=[self.device_id], find_unused_parameters=False)
            self.world_size = dist.get_world_size()
        
        self.opt = torch.optim.AdamW(self._unwrap().parameters(), lr=config["lr"])

        if config.resume_from is not None:
            self._load_state(config.resume_from)

        if config["device"] == "cuda":
            self.unet = torch.compile(self.unet)
            self.ema_unet = torch.compile(self.ema_unet)

        self.dataloader = get_dataloader(config["data_dir"], config["img_size"], config["batch_size"], config["labels_path"], config["num_workers"], config["ddp"])
        self.noise_scheduler = LinearNoiseScheduler(config)

        self.running_meters = {"train/running/mse_loss": RunningMeter(window_size=self.log_every, ddp=self.ddp)}
        self.epoch_meters = {"train/epoch/mse_loss": RunningMeter(window_size=len(self.dataloader) // self.world_size, ddp=self.ddp)}

        logging.info(f"Output DIR: {self.output_dir}")
        logging.info(f"Log every: {self.log_every}")
        logging.info(f"LOG DIR: {self.output_dir / 'logs'}")
        logging.info(f"Train epoches: {self.epochs}")
        logging.info(f"Num images to sample: {self.num_img_to_sample}")
        logging.info(f"CKPT every: {self.ckpt_every}")
        logging.info(f"CFG scale: {self.cfg_scale}")
        logging.info(f"Use weighted loss: {self.use_weighted_loss}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Number of parameters: {human_format(count_parameters(self.unet))}")


    def _unwrap(self):
        if self.ddp:
            return self.unet.module
        return self.unet
    
    def _load_state(self, ckpt_path: Path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self._unwrap().load_state_dict({k.split("_orig_mod.")[1]: p for k, p in ckpt["unet"].items()})
        self.opt.load_state_dict(ckpt["opt"])
        self.last_epoch = ckpt["epoch"]

        logging.info(f"Resume training from {ckpt_path} from last epoch {self.last_epoch}")

        if "unet_ema" in ckpt:
            self.ema_unet.load_state_dict({k.split("_orig_mod.")[1]: p for k, p in ckpt["unet_ema"].items()})
            logging.info(f"Loaded EMA model too!")

    def _save_checkpoint(self, epoch: int):
        if self.rank != 0:
            return
        
        if epoch % self.ckpt_every != 0:
            return

        os.makedirs(self.output_dir / "checkpoints", exist_ok=True)

        torch.save({"unet": self._unwrap().state_dict(),
                    "unet_ema": self.ema_unet.state_dict(),
                    "opt": self.opt.state_dict(),
                    "epoch": epoch
                    }, self.output_dir / "checkpoints" / f"unet_epoch_{epoch}_{(epoch + 1) * len(self.dataloader)}_ema_{self.ema.start_step}.pt")
        logging.info(f"Checkpoint saved ({self.ckpt_every})")

    def sample(self):
        sampled_images = self.noise_scheduler.sample_ema(self.unet, self.ema_unet, self.num_img_to_sample)
        
        def save_images(images, path, **kwargs):
            grid = torchvision.utils.make_grid(images, **kwargs)
            img = (255 * np.clip(grid.permute(1, 2, 0).to('cpu').numpy(), 0.0, 1.0)).astype(np.uint8)
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        save_images(sampled_images["imgs"], "/Users/petrushkovm/Projects/Diffusion/Data/imgs.jpg")
        save_images(sampled_images["imgs_ema"], "/Users/petrushkovm/Projects/Diffusion/Data/imgs_ema.jpg")

    
    def train(self):
        start_epoch = self.last_epoch + 1
        steps_per_epoch = len(self.dataloader)
        for epoch in range(start_epoch, self.epochs):
            if self.ddp:
                self.dataloader.sampler.set_epoch(epoch)

            with tqdm(range(steps_per_epoch), disable=not self.rank == 0) as pbar:
                for i, (x0, y) in enumerate(self.dataloader):
                    global_step = i + steps_per_epoch * epoch
                    x0 = x0.to(self.device)

                    if self.cfg_scale > 0 and np.random.rand() < 0.1:
                        y = None
                    
                    if y is not None:
                        y = y.to(self.device)

                    t = self.noise_scheduler.sample_timesteps(x0.shape[0])
                    xt, noise, sigma_t = self.noise_scheduler.noise_image(x0, t)

                    noise_pred = self.unet(xt, t, y)

                    w = 1.0

                    if self.use_weighted_loss:
                        w = sigma_t ** 2

                    loss = (w * (noise - noise_pred).square()).mean()

                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

                    self.ema.update_ema_model(self.unet, self.ema_unet, global_step)

                    self.running_meters["train/running/mse_loss"].update(loss.detach().cpu().item())
                    self.epoch_meters["train/epoch/mse_loss"].update(loss.detach().cpu().item())

                    if global_step % self.log_every == 0:
                        logs = {tag: meter.compute() for tag, meter in self.running_meters.items()}

                        sampled_images = self.noise_scheduler.sample_ema(self.unet, self.ema_unet, self.num_img_to_sample, self.num_classes, self.cfg_scale)
                        logs.update({title: imgs.detach().cpu() for title, imgs in sampled_images.items()})

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
    # trainer.sample()
    logging.info("Done training!")

    if config.ddp:
        dist.destroy_process_group()
        logging.info("Cleaned up DDP!")


if __name__ == "__main__":
    main()
