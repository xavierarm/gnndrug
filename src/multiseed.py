"""
Multi-seed experiments: 4 models x 3 seeds on scaffold split.
Reports mean +/- std for all metrics.

Usage:
    python3 -u src/multiseed.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP as PyGAttentiveFP
from xgboost import XGBClassifier

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    FEATURES_DIR, SPLITS_DIR, PROCESSED_DIR, RESULTS_DIR,
    evaluate, compute_pos_weight, evaluate_gnn, train_gnn, GCN256,
)

import torch.nn as nn

SEEDS = [42, 123, 456]
BATCH_SIZE = 64


class AttentiveFPModel(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=4,
                 num_timesteps=2, dropout=0.2):
        super().__init__()
        self.model = PyGAttentiveFP(
            in_channels=node_dim, hidden_channels=hidden_dim,
            out_channels=1, edge_dim=edge_dim,
            num_layers=num_layers, num_timesteps=num_timesteps,
            dropout=dropout)

    def forward(self, data):
        return self.model(data.x, data.edge_index, data.edge_attr, data.batch).squeeze(-1)


def load_data():
    X = np.load(FEATURES_DIR / "ecfp_2048.npy")
    graphs = torch.load(FEATURES_DIR / "graphs.pt", weights_only=False)
    df = pd.read_csv(PROCESSED_DIR / "egfr_cleaned.csv")
    y = df["label"].values

    idx = {s: np.load(SPLITS_DIR / f"scaffold_{s}.npy")
           for s in ["train", "valid", "test"]}
    return X, y, graphs, idx


def run_ecfp_model(model_class, model_kwargs, X, y, idx, seed):
    np.random.seed(seed)
    model = model_class(**model_kwargs, random_state=seed)
    model.fit(X[idx["train"]], y[idx["train"]])
    preds = model.predict_proba(X[idx["test"]])[:, 1]
    return evaluate(y[idx["test"]], preds)


def run_gnn_model(model_fn, graphs, idx, device, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader = DataLoader([graphs[i] for i in idx["train"]], batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader([graphs[i] for i in idx["valid"]], batch_size=BATCH_SIZE)
    test_loader = DataLoader([graphs[i] for i in idx["test"]], batch_size=BATCH_SIZE)

    model = model_fn().to(device)
    model = train_gnn(model, train_loader, valid_loader, device, verbose=False)

    _, test_preds, test_labels = evaluate_gnn(model, test_loader, device)
    return evaluate(np.array(test_labels), np.array(test_preds))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, y, graphs, idx = load_data()
    node_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    print(f"Loaded {len(graphs)} graphs, node_dim={node_dim}, edge_dim={edge_dim}\n")

    experiments = {
        "ECFP+RF": lambda seed: run_ecfp_model(
            RandomForestClassifier,
            {"n_estimators": 500, "min_samples_leaf": 5,
             "class_weight": "balanced", "n_jobs": -1},
            X, y, idx, seed),
        "ECFP+XGB": lambda seed: run_ecfp_model(
            XGBClassifier,
            {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.05,
             "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
             "scale_pos_weight": (y[idx["train"]] == 0).sum() / max(y[idx["train"]].sum(), 1),
             "eval_metric": "aucpr", "verbosity": 0},
            X, y, idx, seed),
        "GCN-256": lambda seed: run_gnn_model(
            lambda: GCN256(node_dim), graphs, idx, device, seed),
        "AttentiveFP": lambda seed: run_gnn_model(
            lambda: AttentiveFPModel(node_dim, edge_dim), graphs, idx, device, seed),
    }

    all_rows = []
    for name, run_fn in experiments.items():
        print(f"[{name}] running {len(SEEDS)} seeds...")
        seed_results = []
        for seed in SEEDS:
            metrics = run_fn(seed)
            metrics["seed"] = seed
            metrics["Model"] = name
            seed_results.append(metrics)
            print(f"  seed={seed}: ROC-AUC={metrics['ROC-AUC']:.4f}")
        all_rows.extend(seed_results)

    # Raw results
    df_raw = pd.DataFrame(all_rows)
    df_raw.to_csv(RESULTS_DIR / "multiseed_raw.csv", index=False)

    # Summary: mean +/- std
    metric_cols = ["ROC-AUC", "PR-AUC", "EF1%", "EF5%", "P@50", "P@100"]
    summary_rows = []
    for name in experiments:
        subset = df_raw[df_raw["Model"] == name]
        row = {"Model": name}
        for col in metric_cols:
            vals = subset[col].values
            row[f"{col}_mean"] = vals.mean()
            row[f"{col}_std"] = vals.std()
            row[col] = f"{vals.mean():.4f} +/- {vals.std():.4f}"
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(RESULTS_DIR / "multiseed_performance.csv", index=False)

    print("\n" + "=" * 80)
    print("  MULTI-SEED RESULTS (Scaffold Split, 3 seeds)")
    print("=" * 80)
    for _, row in df_summary.iterrows():
        print(f"  {row['Model']:<15} ROC-AUC: {row['ROC-AUC']}  PR-AUC: {row['PR-AUC']}")

    print(f"\nSaved -> {RESULTS_DIR / 'multiseed_performance.csv'}")


if __name__ == "__main__":
    main()
