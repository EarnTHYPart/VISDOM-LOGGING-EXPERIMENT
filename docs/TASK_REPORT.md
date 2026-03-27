# GSOC Starter Task Report

This document summarizes completion status, implementation details, and reproducibility notes for the 5 required starter tasks.

## Environment

- OS: Windows
- Python: 3.13 (venv)
- Key packages:
  - `torch 2.11.0+cpu`
  - `torchvision 0.26.0+cpu`
  - `pytorch-lightning 2.6.1`
  - `visdom 0.2.4`

## Task 1: Train a simple CNN on MNIST with manual Visdom logging

Status: Completed

Implementation:

- File: `src/train_mnist_visdom.py`
- Model: small CNN (`SimpleCNN`) on MNIST.
- Manual Visdom windows:
  - `Train Loss` (step-wise)
  - `Grad Norm` (step-wise)
  - `Test Accuracy` (epoch-wise)

Run command:

```bash
python src/train_mnist_visdom.py --epochs 1
```

Observed run result:

- Completed successfully.
- Example output: `Epoch 1/1 - loss=0.2311 - test_acc=0.9775`

## Task 2: Install PyTorch Lightning and understand logger interface

Status: Completed

Implementation evidence:

- Lightning training script: `src/train_mnist_lightning.py`
- Custom logger subclass usage: `class VisdomLogger(Logger)` in `src/train_mnist_lightning.py`

Run command:

```bash
python src/train_mnist_lightning.py --epochs 1
```

Observed run result:

- Completed successfully (CPU).
- Metrics were published through the custom logger interface.

## Task 3: Implement a basic custom Lightning logger

Status: Completed

Implementation:

- File: `src/train_mnist_lightning_basic_logger.py`
- Custom logger class: `JsonlLogger(Logger)`
- Behavior:
  - logs hyperparameters
  - logs scalar metrics by step
  - logs finalize status
- Output artifact:
  - `logs/basic_logger/metrics.jsonl`

Run command:

```bash
python src/train_mnist_lightning_basic_logger.py --epochs 1
```

Observed run result:

- Completed successfully.
- JSONL entries generated in `logs/basic_logger/metrics.jsonl`.

## Task 4: Profile hook overhead in PyTorch

Status: Completed

Implementation:

- File: `src/profile_hook_overhead.py`
- Compares average step latency for:
  - no hooks
  - no-op forward hooks
  - lightweight forward hooks

Run command:

```bash
python src/profile_hook_overhead.py --steps 200 --warmup 50
```

Observed run result (CPU, latest full re-run):

- No hooks: `16.3112 ms/step`
- No-op hooks: `141.1451 ms/step`
- Light hooks: `151.1559 ms/step`

## Task 5: Add a gradient norm logger to Visdom client

Status: Completed

Implementation:

- File: `src/train_mnist_visdom.py`
- Added:
  - `grad_norm(model)` helper computing global L2 norm of gradients
  - Visdom `Grad Norm` line window updated each training step

Run command:

```bash
python src/train_mnist_visdom.py --epochs 1
```

Observed run result:

- Completed successfully.
- Visdom showed `Train Loss`, `Grad Norm`, and `Test Accuracy`.

## Artifact Notes

- `data/MNIST/raw/*` files are dataset binaries (IDX and `.gz`).
- `jsonl/` and `visdom/` may contain Lightning checkpoint binaries (`.ckpt`).
- Human-readable custom logger output is in `logs/basic_logger/metrics.jsonl`.

## Reproducibility

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start Visdom server:

```bash
python -m visdom.server -port 8097
```

3. Run task scripts using the commands listed in each section.

## Latest Full Re-Run Evidence (2026-03-27)

Executed scripts:

- `python src/profile_hook_overhead.py --steps 200 --warmup 50`
- `python src/train_mnist_lightning_basic_logger.py --epochs 1`
- `python src/train_mnist_visdom.py --epochs 1`
- `python src/train_mnist_lightning.py --epochs 1`

Observed outputs:

- Manual Visdom script:
  - `Epoch 1/1 - loss=0.2311 - test_acc=0.9775`
- Lightning + Visdom script final progress-bar metrics:
  - `val_loss=0.0627`
  - `val_acc=0.979`
- Basic logger artifact generated:
  - `logs/basic_logger/metrics.jsonl`
