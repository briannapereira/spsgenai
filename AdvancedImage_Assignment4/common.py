from pathlib import Path
import os
import torch
from torchvision import transforms, datasets, utils as vutils

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def cifar10_loaders(batch_size=128, num_workers=4, root='./data'):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    testset  = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_eval)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_grid(tensor, path, nrow=8):
    ensure_dir(os.path.dirname(path))
    vutils.save_image((tensor*0.5 + 0.5).clamp(0,1), path, nrow=nrow)