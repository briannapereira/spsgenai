import torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T
from CNN_Assignment2.model import CNN64
from pathlib import Path

device = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))
print("Using device:", device)

def main():
    tfm_train = T.Compose([T.Resize((64,64)), T.RandomHorizontalFlip(), T.ToTensor()])
    tfm_test  = T.Compose([T.Resize((64,64)), T.ToTensor()])


    PKG_DIR = Path(__file__).parent
    DATA_DIR = PKG_DIR / "data"          
    ckpt_path = PKG_DIR / "cnn64_cifar10.pt"

    trainset = torchvision.datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=tfm_train)
    testset  = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tfm_test)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,  num_workers=0)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=256, shuffle=False, num_workers=0)

    model = CNN64().to(device)
    crit  = nn.CrossEntropyLoss()
    opt   = optim.Adam(model.parameters(), lr=1e-3)

    def eval_once():
        model.eval(); correct=total=0; loss_sum=0.0
        with torch.no_grad():
            for x,y in testloader:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                loss_sum += crit(logits,y).item()*y.size(0)
                correct += (logits.argmax(1)==y).sum().item(); total += y.size(0)
        print(f"[Eval] loss={loss_sum/total:.4f} acc={correct/total:.4f}")
        model.train()

    for ep in range(5):
        run=0.0
        for i,(x,y) in enumerate(trainloader,1):
            x,y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x),y); loss.backward(); opt.step()
            run += loss.item()
            if i%100==0: 
                print(f"[Epoch {ep+1}] iter {i} loss={run/100:.4f}")
                run=0.0
        eval_once()

    ckpt_path = Path(__file__).parent / "cnn64_cifar10.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    print("Saved to", ckpt_path)

if __name__ == "__main__":
    main()