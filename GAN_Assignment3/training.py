import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from GAN_Assignment3.model import Generator, Discriminator, weights_init_dcgan

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Z_DIM = 100
BATCH_SIZE = 128
EPOCHS = 5
LR_G = 2e-4
LR_D = 2e-4
BETAS = (0.5, 0.999)
SAMPLE_DIR = "samples"
CKPT_DIR = "checkpoints"
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])
train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)


G = Generator(Z_DIM).to(DEVICE)
D = Discriminator().to(DEVICE)
G.apply(weights_init_dcgan)
D.apply(weights_init_dcgan)

opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=BETAS)
opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=BETAS)
criterion = nn.BCEWithLogitsLoss()

fixed_z = torch.randn(64, Z_DIM, device=DEVICE)  

#Trainer
def train():
    step = 0
    for epoch in range(1, EPOCHS + 1):
        for imgs, _ in train_loader:
            imgs = imgs.to(DEVICE)

          
            D.train(); G.train()
            bsz = imgs.size(0)
            real_labels = torch.ones(bsz, 1, device=DEVICE)
            fake_labels = torch.zeros(bsz, 1, device=DEVICE)

           
            logits_real = D(imgs)
            loss_real = criterion(logits_real, real_labels)

            
            z = torch.randn(bsz, Z_DIM, device=DEVICE)
            fake_imgs = G(z).detach()
            logits_fake = D(fake_imgs)
            loss_fake = criterion(logits_fake, fake_labels)

            loss_D = loss_real + loss_fake
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            z = torch.randn(bsz, Z_DIM, device=DEVICE)
            gen_imgs = G(z)
            logits_gen = D(gen_imgs)
           
            loss_G = criterion(logits_gen, real_labels)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            if step % 200 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] Step {step} | D: {loss_D.item():.4f} | G: {loss_G.item():.4f}")

        
            if step % 400 == 0:
                G.eval()
                with torch.no_grad():
                    samples = G(fixed_z)
                    
                    utils.save_image(samples, f"{SAMPLE_DIR}/epoch{epoch:03d}_step{step:06d}.png",
                                     nrow=8, normalize=True, value_range=(-1, 1))
            step += 1

        
        torch.save(G.state_dict(), f"{CKPT_DIR}/G_epoch{epoch:03d}.pt")
        torch.save(D.state_dict(), f"{CKPT_DIR}/D_epoch{epoch:03d}.pt")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    train()
    print("Training complete. Check 'samples/' for images and 'checkpoints/' for weights.")