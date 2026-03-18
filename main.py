import argparse
import csv
import itertools
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader, TensorDataset

from model_base import ConvModel
from model_factory import ModelFactory
from tfbind_utils import METRICS_DIR, dna_to_one_hot, load_numpy_dataset
from tf_predict import run_predict

ASSETS_ROOT = "/buildhub/mll/deep_learning_for_bio/assets"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_STEPS = 500
CHECKPOINT_PATH = "/buildhub/mll/deep_learning_for_bio/assets/dna/models"


from tfbind import load_data, preprocess, build_model, train_model, load_checkpoint


def assets(path: str) -> str:
    return os.path.join(ASSETS_ROOT, path)

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
        "--model-id", default="conv2_k10_gelu",
        help="Model ID to use (default: conv2_k10_gelu)",
    )
    parser.add_argument(
        "--steps", type=int, default=NUM_STEPS,
        help=f"Number of training steps (default: {NUM_STEPS})",
    )
    args = parser.parse_args()

    train_csv = assets("dna/datasets/CTCF_train_sequences.csv")
    valid_csv = assets("dna/datasets/CTCF_valid_sequences.csv")

    train_df, valid_ds = load_data(train_csv, valid_csv)
    model, optimizer, criterion = build_model(args.model_id, learning_rate=LEARNING_RATE)

    if args.op == "train":
        train_loader = preprocess(train_df, batch_size=BATCH_SIZE)
        train_model(model, optimizer, criterion, train_loader, num_steps=args.steps)

    elif args.op == "predict":
        checkpoint_path = os.path.join(CHECKPOINT_PATH, f"{args.model_id}.pt")
        if not os.path.exists(checkpoint_path):
            print("model not found", checkpoint_path)
            return
        model = load_checkpoint(model, checkpoint_path)
        run_predict(model, train_csv, valid_csv, model_id=args.model_id)

    elif args.op == "sweep":
        all_model_ids = list(ModelFactory._registry.keys())
        for model_id in all_model_ids:
            print(f"\n=== Sweep: training {model_id} ===")
            model, optimizer, criterion = build_model(model_id, learning_rate=LEARNING_RATE)
            train_loader = preprocess(train_df, batch_size=BATCH_SIZE)
            train_model(model, optimizer, criterion, train_loader, num_steps=args.steps)

            print(f"=== Sweep: predicting {model_id} ===")
            checkpoint_path = os.path.join(CHECKPOINT_PATH, f"{model_id}.pt")
            if not os.path.exists(checkpoint_path):
                print(f"  checkpoint not found after training: {checkpoint_path}, skipping predict")
                continue
            model = load_checkpoint(model, checkpoint_path)
            run_predict(model, train_csv, valid_csv, model_id=model_id)


if __name__ == "__main__":
    main()
