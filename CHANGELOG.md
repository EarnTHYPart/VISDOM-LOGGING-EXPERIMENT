# Changelog

## 2026-03-27

### Updated

- Added a full re-run verification pass across all scripts in `src/` and refreshed documentation:
  - `README.md`
  - `docs/TASK_REPORT.md`
- Corrected GSOC report link in README to `docs/TASK_REPORT.md`.

### Latest Full Re-Run

- `python src/profile_hook_overhead.py --steps 200 --warmup 50`
  - No hooks: `16.3112 ms/step`
  - No-op hooks: `141.1451 ms/step`
  - Light hooks: `151.1559 ms/step`
- `python src/train_mnist_lightning_basic_logger.py --epochs 1`
  - Completed; wrote `logs/basic_logger/metrics.jsonl`
- `python src/train_mnist_visdom.py --epochs 1`
  - Completed; `Epoch 1/1 - loss=0.2311 - test_acc=0.9775`
- `python src/train_mnist_lightning.py --epochs 1`
  - Completed; final progress-bar metrics included `val_loss=0.0627`, `val_acc=0.979`

### Added

- `src/train_mnist_visdom.py`
  - Gradient norm logging to Visdom (`Grad Norm` window, global L2 norm per step).
- `src/train_mnist_lightning_basic_logger.py`
  - Basic custom Lightning logger (`JsonlLogger`) that writes JSONL metrics.
- `src/profile_hook_overhead.py`
  - Benchmark for hook overhead (none vs no-op vs light hook logic).
- README documentation for:
  - Basic custom logger usage
  - Hook profiling usage
  - Verified run notes

### Verification Runs

- `python src/profile_hook_overhead.py --steps 200 --warmup 50`
  - No hooks: `13.9003 ms/step`
  - No-op hooks: `13.8220 ms/step`
  - Light hooks: `15.2150 ms/step`
- `python src/train_mnist_lightning_basic_logger.py --epochs 1`
  - Completed; wrote `logs/basic_logger/metrics.jsonl`
- `python src/train_mnist_visdom.py --epochs 1`
  - Completed; logged `Train Loss`, `Grad Norm`, and `Test Accuracy` to Visdom
- `python src/train_mnist_lightning.py --epochs 1` with Visdom server running
  - Completed; metrics logged to Visdom
