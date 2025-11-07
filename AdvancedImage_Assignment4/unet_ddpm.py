from typing import Optional
import math
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_ch, out_ch))

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time(t_emb)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_ch):
        super().__init__()
        self.block1 = ResidualBlock(in_ch, out_ch, time_ch)
        self.block2 = ResidualBlock(out_ch, out_ch, time_ch)
        self.pool = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)  

    def forward(self, x, t):
        x = self.block1(x, t)
        x = self.block2(x, t)
        skip = x
        x = self.pool(x)
        return x, skip


class Up(nn.Module):
    """Upsample, concatenate skip, then two residual blocks.
       IMPORTANT: we accept skip_ch explicitly to match channels after concat.
    """
    def __init__(self, in_ch, skip_ch, out_ch, time_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block1 = ResidualBlock(out_ch + skip_ch, out_ch, time_ch)
        self.block2 = ResidualBlock(out_ch, out_ch, time_ch)

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t)
        x = self.block2(x, t)
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
       
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class UNet(nn.Module):
    def __init__(self, in_ch=3, base=64, time_dim=128):
        super().__init__()
        self.time = TimeEmbedding(time_dim)
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

  
        self.down1 = Down(base,       base,     time_dim)   
        self.down2 = Down(base,       base * 2, time_dim)   
        self.down3 = Down(base * 2,   base * 4, time_dim)   

        self.mid1 = ResidualBlock(base * 4, base * 4, time_dim)
        self.mid2 = ResidualBlock(base * 4, base * 4, time_dim)

    
        self.up3 = Up(base * 4, base * 4, base * 2, time_dim) 
        self.up2 = Up(base * 2, base * 2, base,     time_dim)  
        self.up1 = Up(base,     base,     base,     time_dim)  

        self.out_norm = nn.GroupNorm(8, base)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time(t)
        x = self.in_conv(x)
        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x, s3 = self.down3(x, t_emb)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        x = self.up3(x, s3, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)

        x = self.out_conv(self.out_act(self.out_norm(x)))
        return x



class DDPM(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int = 1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.T = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x0 + sqrt_om * noise

    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        noise_pred = self.model(xt, t.float())
        return nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, x, t):
     
        beta_t = self.betas[t][:, None, None, None]
        alpha_bar_t = self.alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        eps = self.model(x, t.float())
        mean = (1 / torch.sqrt(self.alphas[t][:, None, None, None])) * (x - eps * (1 - self.alphas[t][:, None, None, None]) / sqrt_one_minus.clamp_min(1e-8))
      
        nonzero = (t > 0).float()[:, None, None, None]
        z = torch.randn_like(x)
        return mean + nonzero * torch.sqrt(beta_t) * z

    @torch.no_grad()
    def sample(self, shape, steps=None, device=None):
        steps = steps or self.T
        device = device or next(self.parameters()).device
        x = torch.randn(shape, device=device)
        for i in reversed(range(steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x.clamp(-1, 1)
