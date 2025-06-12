import os
import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from unet import UNet
from schedulers import LinearNoiseScheduler
from utils import set_seed


def tensor_to_img(x: torch.Tensor):
    img = x.cpu().numpy()
    img = img.transpose(1, 2, 0)
    return np.clip(255 * img, 0, 255).astype(np.uint8)


def save_img(img, save_dir, fname):
    cv2.imwrite(os.path.join(save_dir, f"{fname}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


@hydra.main(config_path="../configs/", config_name="generate_images.yaml", version_base="1.3")
def main(config: DictConfig):
    if config.random_seed is not None:
        set_seed(config.random_seed)

    model = UNet(time_dim=config["time_dim"], width=config["width"], num_classes=config["num_classes"], device=config.device).to(config.device)
    model.eval()
    noise_scheduler = LinearNoiseScheduler(config)

    ckpt = torch.load(config.ckpt_path, map_location="cpu")

    if config.use_ema and "unet_ema" in ckpt:
        model.load_state_dict({k.split("_orig_mod.")[1]: p for k, p in ckpt["unet_ema"].items()})
    else:
        model.load_state_dict({k.split("_orig_mod.")[1]: p for k, p in ckpt["unet"].items()})
    
    print(f"Loaded EMA weights: {config.use_ema}")

    sampled_images = noise_scheduler.sample(model, config.num_img_to_sample, num_classes=config.get("num_classes", None), cfg_scale=config.get("cfg_scale", 0.0))

    suffix = "_ema" if config.use_ema else ""
    for i in range(sampled_images.shape[0]):
        save_img(tensor_to_img(sampled_images[i, ...]), config.output_dir, f"{i}_{config.img_size}{suffix}")


if __name__ == "__main__":
    main()