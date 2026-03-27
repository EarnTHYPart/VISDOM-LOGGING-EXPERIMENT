import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from visdom import Visdom


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--visdom-server", type=str, default="http://localhost")
    parser.add_argument("--visdom-port", type=int, default=8097)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    viz = Visdom(server=args.visdom_server, port=args.visdom_port)
    if not viz.check_connection(timeout_seconds=5):
        raise RuntimeError(
            "Cannot connect to Visdom server. Start it with: python -m visdom.server -port 8097"
        )

    loss_win = None
    acc_win = None
    step = 0

    for epoch in range(args.epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_value = float(loss.item())
            x_value = float(step)

            if loss_win is None:
                loss_win = viz.line(
                    X=[x_value],
                    Y=[loss_value],
                    opts={"title": "Train Loss", "xlabel": "Step", "ylabel": "Loss"},
                )
            else:
                viz.line(X=[x_value], Y=[loss_value], win=loss_win, update="append")

            step += 1

        acc = evaluate(model, test_loader, device)
        mean_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss={mean_loss:.4f} - test_acc={acc:.4f}")

        epoch_x = float(epoch + 1)
        acc_y = float(acc)
        if acc_win is None:
            acc_win = viz.line(
                X=[epoch_x],
                Y=[acc_y],
                opts={"title": "Test Accuracy", "xlabel": "Epoch", "ylabel": "Accuracy"},
            )
        else:
            viz.line(X=[epoch_x], Y=[acc_y], win=acc_win, update="append")


if __name__ == "__main__":
    main()
