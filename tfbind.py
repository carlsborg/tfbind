import argparse
import csv
import itertools
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader, TensorDataset

from constants import timeout
from tfbind_utils import METRICS_DIR, dna_to_one_hot, load_numpy_dataset
from tf_predict import run_predict


# ---------------------------------------------------------------------------
# The Models
# ---------------------------------------------------------------------------

class ConvModelV2(nn.Module):
    """CNN with batch norm and dropout for binary classification."""

    def __init__(
        self,
        conv_filters: int = 64,
        kernel_size: int = 10,
        dense_units: int = 128,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        # First convolutional layer.
        self.conv1 = nn.Conv1d(4, conv_filters, kernel_size=kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Second convolutional layer.
        self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size=kernel_size, padding="same")
        self.bn2 = nn.BatchNorm1d(conv_filters)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Dense layers.
        self.dropout = nn.Dropout(p=dropout_rate)
        self.dense1 = nn.LazyLinear(dense_units)
        self.dense2 = nn.Linear(dense_units, dense_units // 2)
        self.output = nn.Linear(dense_units // 2, 1)

    def model_id(self) -> str:
        return "conv_model_v2"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, seq_len, 4) — convert to (batch, 4, seq_len) for Conv1d.
        x = x.permute(0, 2, 1).float()

        # First convolutional layer.
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.gelu(x)
        x = self.pool1(x)

        # Second convolutional layer.
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.gelu(x)
        x = self.pool2(x)

        # Flatten.
        x = x.flatten(start_dim=1)

        # First dense layer.
        x = self.dense1(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x)

        # Second dense layer.
        x = self.dense2(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x)

        return self.output(x)


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

def build_model(learning_rate: float) -> tuple[nn.Module, torch.optim.Optimizer, nn.BCEWithLogitsLoss]:
    """Instantiate model, Adam optimizer, and BCE loss."""
    model = ConvModelV2()
    for name, param in model.named_parameters():
        print(f"  {name:<35s}")
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
    checkpoint_path: str,
    num_steps: int,
    run_id: str | None = None,
) -> nn.Module:
    """Run the training loop for *timeout* seconds, save a checkpoint, and write loss metrics to CSV.

    The timer starts after a warmup step that absorbs any JIT / compilation
    overhead.  Only actual training steps count toward the time budget.
    """
    os.makedirs(METRICS_DIR, exist_ok=True)
    run_id = run_id or model.model_id()
    loss_csv = os.path.join(METRICS_DIR, f"{run_id}_train_loss.csv")

    model.train()
    train_iter = itertools.cycle(train_loader)

    # --- Warmup step (excluded from the timed budget) ----------------------
    seqs, labels = next(train_iter)
    optimizer.zero_grad()
    logits = model(seqs)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    print(f"  warmup step  loss={loss.item():.4f}")

    # --- Timed training loop -----------------------------------------------
    start_time = time.monotonic()

    with open(loss_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        # Record the warmup step as step 0
        writer.writerow([0, loss.item()])

        step = 1
        while True:
            seqs, labels = next(train_iter)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            writer.writerow([step, loss.item()])
            if step % 100 == 0:
                elapsed = time.monotonic() - start_time
                print(f"  step {step:4d}  loss={loss.item():.4f}  elapsed={elapsed:.1f}s")
            step += 1

            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                print(f"Training time budget reached ({elapsed:.1f}s >= {timeout}s) after {step} steps.")
                break

    print(f"Loss metrics written to {loss_csv}")
    save_checkpoint(model, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    return model


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str) -> nn.Module:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}. Run --op train first.")
    model.load_state_dict(torch.load(path, weights_only=True))
    return model

