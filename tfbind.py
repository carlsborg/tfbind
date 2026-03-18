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

# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_data(train_csv: str, valid_csv: str) -> tuple[pd.DataFrame, dict]:
    """Load raw training DataFrame and validation numpy arrays."""
    train_df = pd.read_csv(train_csv)
    print(train_df)
    print(train_df["label"].value_counts())
    valid_ds = load_numpy_dataset(valid_csv)
    return train_df, valid_ds


def preprocess(train_df: pd.DataFrame, batch_size: int) -> DataLoader:
    """One-hot encode sequences and return a shuffled, cycling DataLoader."""
    sequences = np.array([dna_to_one_hot(seq) for seq in train_df["sequence"]])
    labels = train_df["label"].values[:, None]

    dataset = TensorDataset(
        torch.from_numpy(sequences).float(),
        torch.from_numpy(labels).float(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    sample_seqs, sample_labels = next(iter(loader))
    print(f"Batch sequence shape: {tuple(sample_seqs.shape)}")
    print(f"Batch labels shape:   {tuple(sample_labels.shape)}")

    return loader


def build_model(model_id: str, learning_rate: float) -> tuple[nn.Module, torch.optim.Optimizer, nn.BCEWithLogitsLoss]:
    """Instantiate model, Adam optimizer, and BCE loss."""
    model = ModelFactory.build(model_id)
    for name, param in model.named_parameters():
        print(f"  {name:<35s}  {tuple(param.shape)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    return model, optimizer, criterion


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.BCEWithLogitsLoss,
    train_loader: DataLoader,
    num_steps: int,
) -> ConvModel:
    """Run the full training loop, save a checkpoint, and write loss metrics to CSV."""
    os.makedirs(METRICS_DIR, exist_ok=True)
    loss_csv = os.path.join(METRICS_DIR, f"{model.model_id()}_train_loss.csv")

    model.train()
    train_iter = itertools.cycle(train_loader)

    with open(loss_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        for step in tqdm.tqdm(range(num_steps), desc="Training"):
            seqs, labels = next(train_iter)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            writer.writerow([step, loss.item()])
            if step % 100 == 0:
                print(f"  step {step:4d}  loss={loss.item():.4f}")

    print(f"Loss metrics written to {loss_csv}")
    checkpoint_path = os.path.join(CHECKPOINT_PATH, f"{model.model_id()}.pt")
    save_checkpoint(model, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    return model


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str) -> ConvModel:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}. Run --op train first.")
    model.load_state_dict(torch.load(path, weights_only=True))
    return model

