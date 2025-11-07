import argparse
import torch
import torch.optim as optim
from AdvancedImage_Assignment4.energy import EnergyCNN, nce_loss
from AdvancedImage_Assignment4.common import cifar10_loaders, ensure_dir

def sample_noise_like(x):
    return torch.randn_like(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--device', type=str, default='auto')
    ap.add_argument('--ckpt_dir', type=str, default='./checkpoints/ebm')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.device in ['auto','cuda'] else 'cpu')
    train_loader, _ = cifar10_loaders(batch_size=args.batch_size)

    model = EnergyCNN().to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    ensure_dir(args.ckpt_dir)

    global_step = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        for x, _ in train_loader:
            x = x.to(device)
            x_noise = sample_noise_like(x)
            loss, acc = nce_loss(model, x, x_noise)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if global_step % 100 == 0:
                print(f"[EBM] epoch {epoch} step {global_step} | loss {loss.item():.4f} | nce_acc {acc.item():.3f}")
            global_step += 1
        torch.save(model.state_dict(), f"{args.ckpt_dir}/energy_epoch_{epoch:03d}.pt")

if __name__ == '__main__':
    main()


