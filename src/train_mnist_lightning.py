import argparse
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers.logger import Logger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from visdom import Visdom


class VisdomLogger(Logger):
    def __init__(
        self,
        server: str = "http://localhost",
        port: int = 8097,
        env: str = "main",
        experiment_name: str = "lightning-mnist",
    ) -> None:
        super().__init__()
        self._name = "visdom"
        self._version = "0"
        self._experiment_name = experiment_name
        self._viz = Visdom(server=server, port=port, env=env)
        if not self._viz.check_connection(timeout_seconds=5):
            raise RuntimeError(
                "Cannot connect to Visdom server. Start it with: python -m visdom.server -port 8097"
            )
        self._wins: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def experiment(self) -> str:
        return self._experiment_name

    def log_hyperparams(self, params: Any) -> None:
        self._viz.text(str(params), win="hyperparams")

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        for key, value in metrics.items():
            if key in {"epoch"}:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            try:
                y = float(value)
            except (TypeError, ValueError):
                continue

            if key not in self._wins:
                self._wins[key] = self._viz.line(
                    X=[float(step)],
                    Y=[y],
                    opts={"title": key, "xlabel": "Step", "ylabel": key},
                )
            else:
                self._viz.line(
                    X=[float(step)],
                    Y=[y],
                    win=self._wins[key],
                    update="append",
                )

    def finalize(self, status: str) -> None:
        self._viz.text(f"Run finished with status: {status}", win="status")


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

    def prepare_data(self) -> None:
        datasets.MNIST(root="data", train=True, download=True)
        datasets.MNIST(root="data", train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = datasets.MNIST(root="data", train=True, transform=self.transform)
        self.val_ds = datasets.MNIST(root="data", train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=512, shuffle=False)


class LitMNIST(pl.LightningModule):
    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
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
        return self.classifier(self.features(x))

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--visdom-server", type=str, default="http://localhost")
    parser.add_argument("--visdom-port", type=int, default=8097)
    parser.add_argument("--visdom-env", type=str, default="main")
    args = parser.parse_args()

    model = LitMNIST(lr=args.lr)
    dm = MNISTDataModule(batch_size=args.batch_size)
    logger = VisdomLogger(
        server=args.visdom_server,
        port=args.visdom_port,
        env=args.visdom_env,
        experiment_name="lightning-mnist",
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
