import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import DoubleConv, DownBlock, UpBlock, SelfAttention


class UNet(nn.Module):
    def __init__(self, time_dim: int = 256, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.ipconv = DoubleConv(3, 64)
        self.down1 = DownBlock(64, 128)
        self.att1 = SelfAttention(128, 32, 4) # remove size 
        self.down2 = DownBlock(128, 256)
        self.att2 = SelfAttention(256, 16)
        self.down3 = DownBlock(256, 256)
        self.att3 = SelfAttention(256, 8)

        self.b1 = DoubleConv(256, 512)
        self.b2 = DoubleConv(512, 512)
        self.b3 = DoubleConv(512, 256)

        self.up1 = UpBlock(512, 128)
        self.att4 = SelfAttention(128, 16)
        self.up2 = UpBlock(256, 64)
        self.att5 = SelfAttention(64, 32)
        self.up3 = UpBlock(128, 64)
        self.att6 = SelfAttention(64, 64)
        self.outconv = nn.Conv2d(64, 3, 1, 1, 0)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device, dtype=torch.float) / channels))
        # print(t.shape, inv_freq.shape)

        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.ipconv(x)
        
        x2 = self.down1(x1, t)
        x2 = self.att1(x2)
        
        x3 = self.down2(x2, t)
        x3 = self.att2(x3)

        x4 = self.down3(x3, t)
        x4 = self.att3(x4)

        x4 = self.b1(x4)
        x4 = self.b2(x4)
        x4 = self.b3(x4)

        x = self.up1(x4, x3, t)
        x = self.att4(x)

        x = self.up2(x, x2, t)
        x = self.att5(x)

        x = self.up3(x, x1, t)
        x = self.att6(x)

        return self.outconv(x)


if __name__ == "__main__":
    unet = UNet(time_dim=256, device="mps").to("mps")

    x = torch.rand((6, 3, 64, 64), device="mps")
    t = torch.ones((6,), device="mps")
    y = unet(x, t)

    print(x.shape, " ---> ", y.shape)