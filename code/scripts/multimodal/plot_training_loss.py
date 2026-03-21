#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run dir containing step_metrics.csv")
    parser.add_argument("--output", type=Path, default=None, help="Optional output png path")
    args = parser.parse_args()

    step_csv = args.run_dir / "step_metrics.csv"
    epoch_csv = args.run_dir / "epoch_metrics.csv"
    if not step_csv.exists():
        raise FileNotFoundError(f"Missing {step_csv}")

    step_rows = read_csv_rows(step_csv)
    epoch_rows = read_csv_rows(epoch_csv) if epoch_csv.exists() else []

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    steps = [int(r["step"]) for r in step_rows]
    for key in ["total_loss", "refusal_loss", "harmfulness_loss", "norm_loss"]:
        axes[0].plot(steps, [float(r[key]) for r in step_rows], label=key.replace("_loss", ""))
    axes[0].set_title("Step Loss Curves")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if epoch_rows:
        epochs = [int(r["epoch"]) for r in epoch_rows]
        for key in ["total_loss", "refusal_loss", "harmfulness_loss", "norm_loss"]:
            axes[1].plot(epochs, [float(r[key]) for r in epoch_rows], marker="o", label=key.replace("_loss", ""))
        axes[1].set_title("Epoch Mean Loss Curves")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("mean loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis("off")

    out_path = args.output or (args.run_dir / "loss_curve.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
