import os

import numpy as np
import pandas as pd

METRICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics")


def dna_to_one_hot(dna_sequence: str) -> np.ndarray:
    """Convert DNA into a one-hot encoded format with channel ordering ACGT."""
    base_to_one_hot = {
        "A": (1, 0, 0, 0),
        "C": (0, 1, 0, 0),
        "G": (0, 0, 1, 0),
        "T": (0, 0, 0, 1),
        "N": (1, 1, 1, 1),
    }
    return np.array([base_to_one_hot[base] for base in dna_sequence])


def load_numpy_dataset(csv_path: str) -> dict[str, np.ndarray]:
    """Load a CSV into numpy arrays of one-hot sequences and labels."""
    df = pd.read_csv(csv_path)
    return {
        "labels": df["label"].to_numpy()[:, None],
        "sequences": np.array([dna_to_one_hot(seq) for seq in df["sequence"]]),
    }
