import torch.nn as nn


class EMA:
    def __init__(self, beta: float = 0.995, start_step: int = 2000):
        self.beta = beta
        self.start_step = start_step

    def _update_ema_model(self, model: nn.Module, ema_model: nn.Module, use_ema: bool):
        beta = self.beta if use_ema else 0.0
        for mp, emap in zip(model.parameters(), ema_model.parameters()):
            emap.data = emap.data * beta + (1.0 - beta) * mp.data

    def update_ema_model(self, model: nn.Module, ema_model: nn.Module, step: int):
        self._update_ema_model(model, ema_model, use_ema=step >= self.start_step)