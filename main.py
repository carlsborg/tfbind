import argparse
import csv
import itertools
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, TensorDataset

from tfbind_utils import METRICS_DIR, dna_to_one_hot, load_numpy_dataset
from tf_predict import run_predict


from pathlib import Path
parent_folder = Path(__file__).parent.resolve()

ASSETS_ROOT = parent_folder / "assets"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_STEPS = 500
CHECKPOINT_PATH = parent_folder / "assets/dna/models"

TRANSCRIPTION_FACTORS = [
    "ARID3",
    "ATF2",
    "BACH1",
    "CTCF",
    "ELK1",
    "GABPA",
    "MAX",
    "REST",
    "SRF",
    "ZNF24",
]

from tfbind import load_data, preprocess, build_model, train_model, load_checkpoint


def assets(path: str) -> str:
    return os.path.join(ASSETS_ROOT, path)


# ---------------------------------------------------------------------------
# Per-TF workers (run in parallel)
# ---------------------------------------------------------------------------

def _train_one(tf: str, num_steps: int) -> None:
    print(f"\n=== Training {tf} ===")
    train_csv = assets(f"dna/datasets/{tf}_train_sequences.csv")
    valid_csv = assets(f"dna/datasets/{tf}_valid_sequences.csv")
    train_df, valid_ds = load_data(train_csv, valid_csv)
    model, optimizer, criterion = build_model(learning_rate=LEARNING_RATE)
    run_id = f"{tf}_{model.model_id()}"
    checkpoint_path = os.path.join(CHECKPOINT_PATH, f"{run_id}.pt")
    train_loader = preprocess(train_df, batch_size=BATCH_SIZE)
    train_model(model, optimizer, criterion, train_loader, checkpoint_path, num_steps=num_steps, run_id=run_id)


def _predict_one(tf: str) -> tuple[str, float] | None:
    print(f"\n=== Predicting {tf} ===")
    train_csv = assets(f"dna/datasets/{tf}_train_sequences.csv")
    valid_csv = assets(f"dna/datasets/{tf}_valid_sequences.csv")
    model, _, _ = build_model(learning_rate=LEARNING_RATE)
    run_id = f"{tf}_{model.model_id()}"
    checkpoint_path = os.path.join(CHECKPOINT_PATH, f"{run_id}.pt")
    if not os.path.exists(checkpoint_path):
        print(f"  checkpoint not found: {checkpoint_path}, skipping")
        return None
    model = load_checkpoint(model, checkpoint_path)
    metrics = run_predict(model, train_csv, valid_csv, model_id=run_id, tf=tf)
    return tf, metrics["valid"]["auc"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CTCF binding site classifier — PyTorch (Chapter 3)")
    parser.add_argument(
        "--op", choices=["train", "predict", "sweep"], required=True,
        help="train: fit and save the model  |  predict: load and run predictions  |  sweep: train+predict all models",
    )
    parser.add_argument(
        "--steps", type=int, default=NUM_STEPS,
        help=f"Number of training steps (default: {NUM_STEPS})",
    )
    parser.add_argument(
        "--jobs", type=int, default=-1,
        help="Number of parallel jobs (-1 = all CPUs, default: -1)",
    )
    args = parser.parse_args()

    os.makedirs(ASSETS_ROOT, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    if args.op == "train":
        Parallel(n_jobs=args.jobs, prefer="processes")(
            delayed(_train_one)(tf, args.steps) for tf in TRANSCRIPTION_FACTORS
        )

    elif args.op == "predict":
        results = Parallel(n_jobs=args.jobs, prefer="processes")(
            delayed(_predict_one)(tf) for tf in TRANSCRIPTION_FACTORS
        )
        valid_aucs = {tf: auc for tf, auc in results if tf is not None}

        if valid_aucs:
            mean_auc = float(np.mean(list(valid_aucs.values())))
            summary = {"mean_valid_auc": mean_auc, "per_tf": valid_aucs}
            summary_path = os.path.join(METRICS_DIR, "mean_valid_auc.json")
            os.makedirs(METRICS_DIR, exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n=== Mean validation AUC: {mean_auc:.4f} (saved to {summary_path}) ===")

if __name__ == "__main__":
    main()
