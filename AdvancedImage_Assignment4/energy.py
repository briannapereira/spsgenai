from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyCNN(nn.Module):
    """Small CNN that outputs a scalar energy per image.
    E(x) is unconstrained; lower E means more likely under the model.
    We'll train with NCE (classify data vs noise) using -E(x) as a logit.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(64, 128, 3, padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.SiLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(128, 256, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head = nn.Linear(256, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.flatten(1)
        energy = self.head(h)
        return energy.squeeze(1)




def nce_loss(energy_model: nn.Module, x_data: torch.Tensor, x_noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Noise-Contrastive Estimation style loss.
    We treat -E(x) as the logit for 'data' class (1) vs 'noise' (0).
    Returns (loss, accuracy)
    """
    logits_data = -energy_model(x_data) 
    logits_noise = -energy_model(x_noise) 
    logits = torch.cat([logits_data, logits_noise], dim=0)
    labels = torch.cat([
        torch.ones_like(logits_data),
        torch.zeros_like(logits_noise)
    ], dim=0)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == labels).float().mean()
    return loss, acc