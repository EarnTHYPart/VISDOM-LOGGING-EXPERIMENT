# VISDOM Logging Experiment

This repository is initialized for five starter tasks:

1. Train a simple CNN on MNIST with manual Visdom logging.
2. Install PyTorch Lightning and understand the logger interface.
3. Implement a basic custom Lightning logger.
4. Profile hook overhead in PyTorch.
5. Add a gradient norm logger to the Visdom client.

## Completion Status

- [x] 1. Train a simple CNN on MNIST with Visdom logging (manual)
- [x] 2. Install PyTorch Lightning and understand logger interface
- [x] 3. Implement a basic custom Lightning logger
- [x] 4. Profile hook overhead in PyTorch
- [x] 5. Complete starter task: "Add a gradient norm logger to Visdom client"

GSOC report:

- `docs/TASK_REPORT.md`

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

What it logs to Visdom:

- `Train Loss` per optimization step
- `Grad Norm` (global L2 norm of parameter gradients) per step
- `Test Accuracy` per epoch

## Run: PyTorch Lightning + Custom Visdom Logger

```bash
python src/train_mnist_lightning.py --epochs 3
```

What it does:

- Trains `LitMNIST` with a custom `VisdomLogger`.
- Sends scalar metrics (such as `train_loss`, `val_loss`, `val_acc`) to a Visdom dashboard.
- Requires a running Visdom server at `http://localhost:8097` by default.

## Run: Basic Custom Lightning Logger

This script shows a minimal logger implementation that writes metrics to a JSONL file.

```bash
python src/train_mnist_lightning_basic_logger.py --epochs 1
```

Output file:

- `logs/basic_logger/metrics.jsonl`

What it does:

- Demonstrates a minimal custom Lightning logger (`JsonlLogger`).
- Appends structured JSON records for hyperparameters, metrics, and finalize status.
- Useful as a template before integrating external dashboards.

## Profile: PyTorch Hook Overhead

This benchmark compares average step time across three cases:

- no hooks
- no-op forward hooks
- lightweight forward hooks (single scalar read)

```bash
python src/profile_hook_overhead.py --steps 200 --warmup 50
```

What it does:

- Builds a small conv net and benchmarks forward-pass latency in three modes.
- Reports average milliseconds per step for: no hooks, no-op hooks, and light logging hooks.
- Helps estimate instrumentation cost before adding heavy hook-based logging.

## Verified Run Notes (2026-03-27)

Commands run:

- `python src/profile_hook_overhead.py --steps 200 --warmup 50`
- `python src/train_mnist_lightning_basic_logger.py --epochs 1`
- `python -m visdom.server -port 8097`
- `python src/train_mnist_visdom.py --epochs 1`
- `python src/train_mnist_lightning.py --epochs 1`

Observed outcomes:

- Hook profiling (CPU):
	- No hooks: `13.9003 ms/step`
	- No-op hooks: `13.8220 ms/step` (`-0.56%`)
	- Light hooks: `15.2150 ms/step` (`+9.46%`)
- Basic logger run completed and wrote metrics to `logs/basic_logger/metrics.jsonl`.
- Manual Visdom run completed and logged `Train Loss`, `Grad Norm`, and `Test Accuracy`.
- Visdom logger run completed for 1 epoch and published metrics to Visdom session windows.
- Lightning emitted common warnings about low `num_workers` and existing checkpoint directories; runs still completed.

## Latest Full Re-Run (2026-03-27)

All scripts in `src/` were executed again:

- `python src/profile_hook_overhead.py --steps 200 --warmup 50`
- `python src/train_mnist_lightning_basic_logger.py --epochs 1`
- `python src/train_mnist_visdom.py --epochs 1`
- `python src/train_mnist_lightning.py --epochs 1`

Observed outcomes from this full re-run:

- Hook profiling (CPU):
	- No hooks: `16.3112 ms/step`
	- No-op hooks: `141.1451 ms/step` (`+765.33%`)
	- Light hooks: `151.1559 ms/step` (`+826.70%`)
- Basic logger script completed and wrote `logs/basic_logger/metrics.jsonl`.
- Manual Visdom script completed with: `Epoch 1/1 - loss=0.2311 - test_acc=0.9775`.
- Lightning+Visdom script completed with final progress-bar metrics including: `val_loss=0.0627`, `val_acc=0.979`.

## Notes

- MNIST data is downloaded under `data/`.
- If Torch install fails, install a matching wheel from the official PyTorch install selector for your CUDA/CPU setup, then re-run `pip install -r requirements.txt`.
