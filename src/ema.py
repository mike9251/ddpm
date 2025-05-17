import torch


class EMA:
    def __init__(self, beta: float = 0.995, step: int = 0, start_step: int = 2000):
        self.beta = beta
        self.step = step
        self.start_step = start_step
    
    def _reset_ema_model(self, model: torch.nn.Module, ema_model: torch.nn.Module):
        ema_model.load_state_dict(model.state_dict())

    def _update_ema_model(self, model: torch.nn.Module, ema_model: torch.nn.Module):
        for mp, emap in zip(model.parameters(), ema_model.parameters()):
            emap.data = emap.data * self.beta + (1.0 - self.beta) * mp.data

    def update_ema_model(self, model: torch.nn.Module, ema_model: torch.nn.Module):
        if self.step < self.start_step:
            self._reset_ema_model(model, ema_model)
        else:
            self._update_ema_model(model, ema_model)
        
        self.step += 1