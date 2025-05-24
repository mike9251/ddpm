import torch
import torch.nn as nn

from tqdm import tqdm


class LinearNoiseScheduler:
    def __init__(self, config):
        self.noise_steps = config["noise_steps"]
        self.beta_start = config["beta_start"]
        self.beta_end = config["beta_end"]
        self.img_size = config["img_size"]
        self.device = config["device"]

        self.beta = self.prepare_noise_schedule() # noise_steps
        self.alpha = 1.0 - self.beta # noise_steps
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) # noise_steps
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps, device=self.device)
    
    def noise_image(self, x: torch.Tensor, t: torch.Tensor):
        assert x.shape[0] == t.shape[0]

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(x.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t]).view(x.shape[0], 1, 1, 1)
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise
    
    def sample_timesteps(self, n: int) -> torch.Tensor:
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)
    
    @torch.no_grad()
    def sample(self, model: nn.Module, num_images: int):
        model.eval()

        x = torch.randn((num_images, 3, self.img_size, self.img_size), device=self.device)
        for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
            t = i * torch.ones((num_images,), device=self.device, dtype=torch.long)
            pred_noise = model(x, t)
            alpha = self.alpha[t].view(num_images, 1, 1, 1)
            alpha_hat = self.alpha_hat[t].view(num_images, 1, 1, 1)
            beta = self.beta[t].view(num_images, 1, 1, 1)

            noise = torch.randn_like(x)

            if i == 1:
                noise *= 0

            x = 1.0 / torch.sqrt(alpha) * (x - ((1.0 - alpha) / (torch.sqrt(1.0 - alpha_hat))) * pred_noise) + torch.sqrt(beta) * noise

        x = (x.clamp(-1, 1) + 1) / 2

        model.train()

        return x
    
    @torch.no_grad()
    def sample_ema(self, model: nn.Module, ema_model: nn.Module, num_images: int, num_classes: int = 0, cfg_scale: float = 3.0):
        model.eval()

        x = torch.randn((num_images, 3, self.img_size, self.img_size), device=self.device)
        x_ema = x.clone()
        
        y = None
        if num_classes > 0:
            y = []
            img_per_class = num_images // num_classes
            for cl in range(num_classes):
                y.append(cl * torch.ones((img_per_class, )))
            
            y = torch.cat(y, dim=0)
            y = y.int()
            y = y.to(self.device)

        for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
            t = i * torch.ones((num_images,), device=self.device, dtype=torch.long)

            pred_noise = model(x, t, y)
            pred_noise_ema = ema_model(x_ema, t, y)

            if cfg_scale > 0:
                pred_noise_uncond = model(x, t, None)
                pred_noise_ema_uncond = ema_model(x_ema, t, None)
                
                pred_noise = torch.lerp(pred_noise_uncond, pred_noise, cfg_scale)
                pred_noise_ema = torch.lerp(pred_noise_ema_uncond, pred_noise_ema, cfg_scale)

            alpha = self.alpha[t].view(num_images, 1, 1, 1)
            alpha_hat = self.alpha_hat[t].view(num_images, 1, 1, 1)
            beta = self.beta[t].view(num_images, 1, 1, 1)

            noise = torch.randn_like(x)

            if i == 1:
                noise *= 0

            x = 1.0 / torch.sqrt(alpha) * (x - ((1.0 - alpha) / (torch.sqrt(1.0 - alpha_hat))) * pred_noise) + torch.sqrt(beta) * noise
            x_ema = 1.0 / torch.sqrt(alpha) * (x_ema - ((1.0 - alpha) / (torch.sqrt(1.0 - alpha_hat))) * pred_noise_ema) + torch.sqrt(beta) * noise

        x = (x.clamp(-1, 1) + 1) / 2
        x_ema = (x_ema.clamp(-1, 1) + 1) / 2

        model.train()

        return {"imgs": x, "imgs_ema": x_ema}


if __name__ == "__main__":
    device = "mps"

    noise_scheduler = LinearNoiseScheduler({"noise_steps": 1000, "beta_start": 1e-4, "beta_end": 2e-2, "img_size": 64, "device": device})
    print(f"{noise_scheduler.beta.shape=}")

    from dataloader import get_dataloader

    data = get_dataloader("/Users/petrushkovm/Downloads/celeba_hq_256", img_size=64, batch_size=4)

    t = torch.randint(1, 1000, (4,), device=device)

    x0 = next(iter(data))
    print(f"{x0.shape=}")

    x0 = x0.to(device)

    xt, noise = noise_scheduler.noise_image(x0, t)
    print(f"{xt.shape=}, {noise.shape=}")

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x, t):
            return x
    

    x_sample = noise_scheduler.sample(DummyModel(), 12)
    print(f"{x_sample.shape=}")
