import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import DoubleConv, DownBlock, UpBlock, SelfAttention


class UNet(nn.Module):
    def __init__(self, time_dim: int = 256, width: int = 1, num_classes: int = None, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.ipconv = DoubleConv(3, width*64)
        self.down1 = DownBlock(width*64, width*128)
        self.att1 = SelfAttention(width*128, 32, 4) # remove size 
        self.down2 = DownBlock(width*128, width*256)
        self.att2 = SelfAttention(width*256, 16)
        self.down3 = DownBlock(width*256, width*256)
        self.att3 = SelfAttention(width*256, 8)

        self.b1 = DoubleConv(width*256, width*512)
        self.b2 = DoubleConv(width*512, width*512)
        self.b3 = DoubleConv(width*512, width*256)

        self.up1 = UpBlock(width*512, width*128)
        self.att4 = SelfAttention(width*128, 16)
        self.up2 = UpBlock(width*256, width*64)
        self.att5 = SelfAttention(width*64, 32)
        self.up3 = UpBlock(width*128, width*64)
        self.att6 = SelfAttention(width*64, width*64)
        self.outconv = nn.Conv2d(width*64, 3, 1, 1, 0)

        self.cond = num_classes is not None
        if self.cond:
            self.cond_embed = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device, dtype=torch.float) / channels))
        # print(t.shape, inv_freq.shape)

        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if self.cond and cond is not None:
            t += self.cond_embed(cond)

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