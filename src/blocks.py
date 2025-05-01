import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, residual: bool = False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(1, mid_channels), # replace with LayerNorm
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(1, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 256):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels)
        )

        self.emb_layer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.block(x)
        b, c, h, w = x.shape
        emb = self.emb_layer(t).view(b, c, 1, 1).repeat(1, 1, h, w)
        return x + emb


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 256):
        super().__init__()
        self.block = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2)
        )

        self.emb_layer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )
    
    def forward(self, x: torch.Tensor, skip_x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=True)
        x = torch.cat([skip_x, x], dim=1)
        x = self.block(x)

        b, c, h, w = x.shape
        emb = self.emb_layer(t).view(b, c, 1, 1).repeat(1, 1, h, w)
        return x + emb
    

class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, size: int, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.size = size
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(-1, self.embed_dim, h * w).permute(0, 2, 1)
        x_ln = self.ln(x)

        attn, _ = self.attn(x_ln, x_ln, x_ln)
        attn = attn + x
        attn = self.ff(attn) + attn
        return attn.permute(0, 2, 1).view(-1, self.embed_dim, h, w)
    


if __name__ == "__main__":
    x = torch.rand((4, 128, 32, 32))

    dconv1 = DoubleConv(128, 256, 64, False)
    y = dconv1(x)
    print(x.shape, " ---> ", y.shape)

    dconv2 = DoubleConv(128, 128, 64, True)
    y = dconv2(x)
    print(x.shape, " ---> ", y.shape)

    t = torch.rand((4, 256))
    dblock = DownBlock(128, 256)
    y = dblock(x, t)
    print(x.shape, " ---> ", y.shape)

    skip = torch.rand((4, 128, 64, 64))
    upblock = UpBlock(256, 128)
    y = upblock(x, skip, t)
    print(x.shape, " ---> ", y.shape)

    attn = SelfAttention(128, 32, 4)

    y = attn(x)
    print(y.shape)
