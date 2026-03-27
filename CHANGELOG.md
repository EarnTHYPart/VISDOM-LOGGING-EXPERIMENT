# Changelog

## 2026-03-27

### Added

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
- `python src/train_mnist_lightning.py --epochs 1` with Visdom server running
  - Completed; metrics logged to Visdom
