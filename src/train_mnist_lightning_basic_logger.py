import argparse
import json
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers.logger import Logger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class JsonlLogger(Logger):
    """A basic Lightning logger that writes one JSON object per metric event."""

    def __init__(self, log_dir: str = "logs/basic_logger", run_name: str = "mnist-basic") -> None:
        super().__init__()
        self._name = "jsonl"
        self._version = "0"
        self._run_name = run_name
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_file = self._log_dir / "metrics.jsonl"
        self._fp = self._metrics_file.open("a", encoding="utf-8")

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def experiment(self) -> str:
        return self._run_name

    def log_hyperparams(self, params: Any) -> None:
        record = {"type": "hyperparams", "params": str(params)}
        self._fp.write(json.dumps(record) + "\n")
        self._fp.flush()

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        serializable: dict[str, float] = {}
        for key, value in metrics.items():
            if key == "epoch":
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            try:
                serializable[key] = float(value)
            except (TypeError, ValueError):
                continue

        if not serializable:
            return

        record = {"type": "metrics", "step": int(step), "metrics": serializable}
        self._fp.write(json.dumps(record) + "\n")
        self._fp.flush()

    def finalize(self, status: str) -> None:
        record = {"type": "finalize", "status": status}
        self._fp.write(json.dumps(record) + "\n")
        self._fp.flush()
        self._fp.close()


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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-dir", type=str, default="logs/basic_logger")
    args = parser.parse_args()

    model = LitMNIST(lr=args.lr)
    dm = MNISTDataModule(batch_size=args.batch_size)
    logger = JsonlLogger(log_dir=args.log_dir)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=dm)
    print(f"Metrics written to: {Path(args.log_dir) / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()
