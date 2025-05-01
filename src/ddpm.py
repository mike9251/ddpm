import torch
import torch.nn as nn
from tqdm import tqdm

import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s", level=logging.INFO, default="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2, img_size: int = 64, device: str = "cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_image(self, x: torch.Tensor, t: torch.Tensor):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] # b, 1, 1, 1
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise
    
    def sample_timesteps(self, n: int) -> torch.Tensor:
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    @torch.no_grad()
    def sample(self, model: nn.Module, num_images: int):
        model.eval()

        x = torch.randn((num_images, 3, self.img_size, self.img_size), device=self.device)
        for i in tqdm(reversed(range(1, self.noise_steps))):
            t = i * torch.ones((num_images,), device=self.device)
            pred_noise = model(x, t)
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]

            noise = torch.ones_like(x)

            if i == 1:
                noise *= 0
            
            x = 1.0 / torch.sqrt(alpha) * (x - ((1.0 - alpha) / (torch.sqrt(1.0 - alpha_hat))) * pred_noise) + torch.sqrt(beta) * noise

        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
