import argparse
import time

import torch
import torch.nn as nn


class TinyConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def benchmark(
    model: nn.Module,
    x: torch.Tensor,
    steps: int,
    warmup: int,
    use_autocast: bool,
    hook_mode: str,
) -> float:
    handles: list[torch.utils.hooks.RemovableHandle] = []

    if hook_mode == "noop":
        def _noop_hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            return None

        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                handles.append(m.register_forward_hook(_noop_hook))

    if hook_mode == "light":
        sink: list[float] = []

        def _light_hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            # Read one scalar to emulate a lightweight logging action.
            sink.append(float(output.detach().flatten()[0]))

        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                handles.append(m.register_forward_hook(_light_hook))

    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)

    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        with torch.no_grad():
            _ = model(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    for handle in handles:
        handle.remove()

    return (elapsed / steps) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = TinyConvNet().to(device).eval()
    x = torch.randn(args.batch_size, 1, 28, 28, device=device)

    baseline_ms = benchmark(model, x, args.steps, args.warmup, use_autocast=False, hook_mode="none")
    noop_ms = benchmark(model, x, args.steps, args.warmup, use_autocast=False, hook_mode="noop")
    light_ms = benchmark(model, x, args.steps, args.warmup, use_autocast=False, hook_mode="light")

    def _overhead(case_ms: float, base_ms: float) -> float:
        return ((case_ms - base_ms) / base_ms) * 100.0 if base_ms > 0 else 0.0

    print(f"Device: {device}")
    print(f"Steps: {args.steps}, Warmup: {args.warmup}, Batch size: {args.batch_size}")
    print(f"No hooks      : {baseline_ms:.4f} ms/step")
    print(f"No-op hooks   : {noop_ms:.4f} ms/step ({_overhead(noop_ms, baseline_ms):+.2f}% overhead)")
    print(f"Light hooks   : {light_ms:.4f} ms/step ({_overhead(light_ms, baseline_ms):+.2f}% overhead)")


if __name__ == "__main__":
    main()
