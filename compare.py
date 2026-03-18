import glob
import json
import os

import matplotlib.pyplot as plt

from tfbind_utils import METRICS_DIR


def plot_model_metrics(metrics_dir: str = METRICS_DIR, save_path: str | None = None) -> None:
    """Load all *_metrics.json files and render a grouped bar chart comparing models."""
    paths = sorted(glob.glob(os.path.join(metrics_dir, "*_metrics.json")))
    if not paths:
        print(f"No metrics JSON files found in {metrics_dir}")
        return

    records = []
    for p in paths:
        with open(p) as f:
            records.append(json.load(f))

    model_ids = [r["model_id"] for r in records]
    metrics = ["loss", "accuracy", "auc"]
    splits = ["train", "valid"]
    colors = {"train": "#4C72B0", "valid": "#DD8452"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    fig.suptitle("Model comparison", fontsize=14, fontweight="bold")

    x = range(len(model_ids))
    bar_width = 0.35

    for ax, metric in zip(axes, metrics):
        for i, split in enumerate(splits):
            values = [r[split][metric] for r in records]
            offsets = [xi + (i - 0.5) * bar_width for xi in x]
            ax.bar(offsets, values, width=bar_width, label=split, color=colors[split])

        ax.set_title(metric)
        ax.set_xticks(list(x))
        ax.set_xticklabels(model_ids, rotation=45, ha="right", fontsize=8)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def plot_loss_curves(metrics_dir: str = METRICS_DIR, save_path: str | None = None) -> None:
    """Plot training loss curves from *_train_loss.csv — one subplot per model."""
    import csv

    paths = sorted(glob.glob(os.path.join(metrics_dir, "*_train_loss.csv")))
    if not paths:
        print(f"No train loss CSV files found in {metrics_dir}")
        return

    n = len(paths)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle("Training loss curves", fontsize=14, fontweight="bold")

    for ax, path in zip(axes.flat, paths):
        model_id = os.path.basename(path).replace("_train_loss.csv", "")
        steps, losses = [], []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))

        ax.plot(steps, losses, linewidth=1.2)
        ax.set_title(model_id, fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(linestyle="--", alpha=0.5)

    # hide any unused subplots
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    plot_model_metrics()
    plot_loss_curves()
