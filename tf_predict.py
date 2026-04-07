import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from tfbind_utils import dna_to_one_hot, load_numpy_dataset, METRICS_DIR

SANITY_SEQS = {
    "ctcf_motif":  "CCACCAGGGGGCGC" * 14 + "AAAA",
    "all_A":       "A" * 200,
    "all_C":       "C" * 200,
    "all_G":       "G" * 200,
    "all_T":       "T" * 200,
    "ACGT_repeat": "ACGTACGT" * 25,
    "TCGA_repeat": "TCGATCGT" * 25,
    "TATA_repeat": "TATACGCG" * 25,
    "CAGG_repeat": "CAGGCAGG" * 25,
}


def predict_sanity_check(model: nn.Module, tf: str | None = None) -> None:
    """Predict on random control sequences, and CTCF motif if tf=='CTCF'."""
    model.eval()
    print("\n--- Sanity check ---")
    seqs = {k: v for k, v in SANITY_SEQS.items() if k != "ctcf_motif"}
    if tf == "CTCF":
        seqs = SANITY_SEQS
    with torch.no_grad():
        for name, seq in seqs.items():
            x = torch.from_numpy(dna_to_one_hot(seq)[None, :]).float()
            prob = float(torch.sigmoid(model(x))[0, 0])
            print(f"  {name:<20s}  prob={prob:.4f}")


def predict_on_dataset(
    model: nn.Module, dataset: dict, name: str,
    batch_size: int = 512, save_path: str | None = None,
) -> dict:
    """Compute loss, accuracy, and AUC over a full numpy dataset."""
    model.eval()
    n = len(dataset["labels"])
    all_logits = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            seqs = torch.from_numpy(dataset["sequences"][i : i + batch_size]).float()
            all_logits.append(model(seqs).numpy())

    logits = np.concatenate(all_logits)
    labels = dataset["labels"]

    loss = float(nn.BCEWithLogitsLoss()(
        torch.tensor(logits), torch.tensor(labels).float()
    ))
    accuracy = float(((torch.sigmoid(torch.tensor(logits)) >= 0.5).numpy() == labels).mean())
    auc = roc_auc_score(labels, logits)

    print(f"  [{name:<6s}]  n={n}  loss={loss:.4f}  accuracy={accuracy:.4f}  auc={auc:.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, logits=logits, labels=labels)
        print(f"  Predictions saved to {save_path}")

    return {"n": n, "loss": loss, "accuracy": accuracy, "auc": auc}


def run_predict(model: nn.Module, train_csv: str, valid_csv: str, model_id: str = "model", tf: str | None = None) -> None:
    """Run predictions against sanity check, train, and validation datasets."""
    predict_sanity_check(model, tf=tf)

    print("\n--- Dataset predictions ---")
    train_metrics = predict_on_dataset(model, load_numpy_dataset(train_csv), "train")
    valid_metrics = predict_on_dataset(
        model, load_numpy_dataset(valid_csv), "valid",
        save_path=os.path.join(METRICS_DIR, f"{model_id}_val_preds.npz"),
    )

    metrics = {"model_id": model_id, "train": train_metrics, "valid": valid_metrics}
    metrics_path = os.path.join(METRICS_DIR, f"{model_id}_metrics.json")
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")
    return metrics
