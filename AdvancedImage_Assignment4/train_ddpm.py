import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from AdvancedImage_Assignment4.unet_ddpm import UNet, DDPM
from AdvancedImage_Assignment4.common import ensure_dir, save_grid


def cifar10_loaders_ddpm(batch_size=128, num_workers=2, root='./data'):
    """
    For DDPM we want inputs scaled to [-1, 1]. Using Normalize(mean=0.5, std=0.5)
    on each channel is equivalent to x -> 2x - 1, and is picklable (no lambda).
    """
    tx = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  
    ])
    trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=tx)
    return DataLoader(trainset,
                  batch_size=batch_size,
                  shuffle=True,
                  num_workers=0, 
                  pin_memory=True), None


def pick_device(pref: str):
    pref = pref.lower()
    if pref in ("cuda", "gpu") and torch.cuda.is_available():
        return torch.device("cuda")
    if pref in ("mps",) and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref in ("cpu",):
        return torch.device("cpu")
  
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--timesteps', type=int, default=1000)
    ap.add_argument('--device', type=str, default='auto')   
    ap.add_argument('--ckpt_dir', type=str, default='./checkpoints/ddpm')
    ap.add_argument('--sample_steps', type=int, default=250)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[DDPM] Using device: {device}")

    train_loader, _ = cifar10_loaders_ddpm(batch_size=args.batch_size)

    unet = UNet()
    ddpm = DDPM(unet, timesteps=args.timesteps).to(device)
    opt = optim.AdamW(ddpm.parameters(), lr=args.lr)


    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ensure_dir(args.ckpt_dir)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        ddpm.train()
        for x, _ in train_loader:
            x = x.to(device)
            b = x.size(0)
            t = torch.randint(0, ddpm.T, (b,), device=device, dtype=torch.long)

            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = ddpm.p_losses(x, t)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss = ddpm.p_losses(x, t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
                opt.step()

            if global_step % 100 == 0:
                print(f"[DDPM] epoch {epoch} step {global_step} | loss {loss.item():.4f}")
            global_step += 1

        
        ckpt_path = f"{args.ckpt_dir}/ddpm_epoch_{epoch:03d}.pt"
        torch.save(ddpm.state_dict(), ckpt_path)
        print(f"[DDPM] saved {ckpt_path}")

      
        ddpm.eval()
        with torch.no_grad():
            imgs = ddpm.sample((16, 3, 32, 32), steps=args.sample_steps, device=device)
        
            img_for_save = imgs * 0.5 + 0.5
            save_grid(img_for_save, f"{args.ckpt_dir}/samples_epoch_{epoch:03d}.png", nrow=4)

    print("[DDPM] Done.")


if __name__ == '__main__':
    main()
