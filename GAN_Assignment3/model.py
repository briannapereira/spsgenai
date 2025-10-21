import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Input: z ~ (B, 100)
    FC -> (B, 128*7*7) -> view (B, 128, 7, 7)
    ConvT: 128->64, k4,s2,p1 -> (B, 64, 14, 14) + BN + ReLU
    ConvT: 64->1,  k4,s2,p1 -> (B, 1, 28, 28)  + Tanh
    """
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),

        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),  
        )

    def forward(self, z):
        x = self.net(z)
        x = x.view(z.size(0), 128, 7, 7)
        x = self.deconv(x)
        return x


class Discriminator(nn.Module):
    """
    Input: (B, 1, 28, 28)
    Conv: 1->64,  k4,s2,p1 -> 14x14 + LeakyReLU(0.2)
    Conv: 64->128,k4,s2,p1 -> 7x7   + BN + LeakyReLU(0.2)
    Flatten -> Linear(128*7*7 -> 1) -> (logit)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(128 * 7 * 7, 1)  

    def forward(self, x):
        h = self.features(x)
        h = h.view(x.size(0), -1)
        logit = self.classifier(h)
        return logit


def weights_init_dcgan(m):
    """Initialize as in DCGAN: N(0, 0.02) for conv/convT/linear & BN gamma ~ N(1,0.02)"""
    classname = m.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.zeros_(m.bias.data)