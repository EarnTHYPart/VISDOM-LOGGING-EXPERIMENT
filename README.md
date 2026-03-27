# VISDOM Logging Experiment

This repository is initialized for two learning goals:

1. Train a simple CNN on MNIST with manual Visdom logging.
2. Train the same idea in PyTorch Lightning and inspect the logger interface.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Visdom server in a separate terminal:

```bash
python -m visdom.server -port 8097
```

4. Open the dashboard:

`http://localhost:8097`

## Run: Manual Visdom Logging

```bash
python src/train_mnist_visdom.py --epochs 3
```

## Run: PyTorch Lightning + Custom Visdom Logger

```bash
python src/train_mnist_lightning.py --epochs 3
```

## Notes

- MNIST data is downloaded under `data/`.
- If Torch install fails, install a matching wheel from the official PyTorch install selector for your CUDA/CPU setup, then re-run `pip install -r requirements.txt`.
