"""
Baseline 모델: ECFP + XGBoost / Random Forest / MLP.
3 models x 2 splits = 6 experiments.

Usage:
    python src/baseline.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "data" / "features"
SPLITS_DIR = ROOT / "data" / "splits"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── 평가 메트릭 ──────────────────────────────────────────────────────

def enrichment_factor(y_true, y_pred_proba, percentage):
    n = len(y_true)
    n_top = max(1, int(n * percentage / 100))
    sorted_idx = np.argsort(y_pred_proba)[::-1]
    top_actives = y_true[sorted_idx[:n_top]].sum()
    total_actives = y_true.sum()
    if total_actives == 0:
        return 0.0
    return (top_actives / n_top) / (total_actives / n)


def top_k_precision(y_true, y_pred_proba, k):
    sorted_idx = np.argsort(y_pred_proba)[::-1]
    return y_true[sorted_idx[:k]].sum() / k


def evaluate(y_true, y_pred_proba):
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    return {
        "ROC-AUC": roc_auc_score(y_true, y_pred_proba),
        "PR-AUC": average_precision_score(y_true, y_pred_proba),
        "EF1%": enrichment_factor(y_true, y_pred_proba, 1),
        "EF5%": enrichment_factor(y_true, y_pred_proba, 5),
        "P@50": top_k_precision(y_true, y_pred_proba, 50),
        "P@100": top_k_precision(y_true, y_pred_proba, 100),
    }


# ── 데이터 로딩 ──────────────────────────────────────────────────────

def load_data(split_name):
    X = np.load(FEATURES_DIR / "ecfp_2048.npy")
    y = pd.read_csv(PROCESSED_DIR / "egfr_cleaned.csv")["label"].values

    train_idx = np.load(SPLITS_DIR / f"{split_name}_train.npy")
    valid_idx = np.load(SPLITS_DIR / f"{split_name}_valid.npy")
    test_idx = np.load(SPLITS_DIR / f"{split_name}_test.npy")

    return (
        X[train_idx], y[train_idx],
        X[valid_idx], y[valid_idx],
        X[test_idx], y[test_idx],
    )


# ── XGBoost ───────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_valid, y_valid):
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=n_neg / max(n_pos, 1),
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=SEED,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model


# ── Random Forest ─────────────────────────────────────────────────────

def train_rf(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(X_train, y_train)
    return model


# ── MLP ───────────────────────────────────────────────────────────────

class FingerprintMLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=(512, 256, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_valid, y_valid, device):
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_va = torch.tensor(X_valid, dtype=torch.float32)
    y_va = torch.tensor(y_valid, dtype=torch.float32)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = FingerprintMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
    )

    best_auc = 0
    patience_counter = 0
    best_state = None

    for epoch in range(200):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # validate
        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(X_va.to(device))).cpu().numpy()
        auc = roc_auc_score(y_valid, preds)
        scheduler.step(auc)

        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

    model.load_state_dict(best_state)
    return model


def predict_mlp(model, X, device):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        return torch.sigmoid(model(X_t)).cpu().numpy()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")  # MPS can segfault with BatchNorm; CPU is fast enough for MLP
    print(f"Device: {device}\n")

    all_results = []

    for split_name in ["random", "scaffold"]:
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(split_name)
        print(f"{'=' * 55}")
        print(f"  Split: {split_name}")
        print(f"  Train: {len(X_train)}  Valid: {len(X_valid)}  Test: {len(X_test)}")
        print(f"{'=' * 55}")

        # XGBoost
        print("  [XGBoost] training...", end=" ", flush=True)
        xgb_model = train_xgboost(X_train, y_train, X_valid, y_valid)
        xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
        xgb_metrics = evaluate(y_test, xgb_preds)
        xgb_metrics.update({"Model": "ECFP+XGB", "Split": split_name})
        all_results.append(xgb_metrics)
        print(f"ROC-AUC={xgb_metrics['ROC-AUC']:.4f}  PR-AUC={xgb_metrics['PR-AUC']:.4f}")

        # Random Forest
        print("  [RF]      training...", end=" ", flush=True)
        rf_model = train_rf(X_train, y_train)
        rf_preds = rf_model.predict_proba(X_test)[:, 1]
        rf_metrics = evaluate(y_test, rf_preds)
        rf_metrics.update({"Model": "ECFP+RF", "Split": split_name})
        all_results.append(rf_metrics)
        print(f"ROC-AUC={rf_metrics['ROC-AUC']:.4f}  PR-AUC={rf_metrics['PR-AUC']:.4f}")

        # MLP
        print("  [MLP]     training...", end=" ", flush=True)
        mlp_model = train_mlp(X_train, y_train, X_valid, y_valid, device)
        mlp_preds = predict_mlp(mlp_model, X_test, device)
        mlp_metrics = evaluate(y_test, mlp_preds)
        mlp_metrics.update({"Model": "ECFP+MLP", "Split": split_name})
        all_results.append(mlp_metrics)
        print(f"ROC-AUC={mlp_metrics['ROC-AUC']:.4f}  PR-AUC={mlp_metrics['PR-AUC']:.4f}")

        print()

    # 결과 정리
    df_results = pd.DataFrame(all_results)
    col_order = ["Model", "Split", "ROC-AUC", "PR-AUC", "EF1%", "EF5%", "P@50", "P@100"]
    df_results = df_results[col_order]

    out_path = RESULTS_DIR / "baseline_performance.csv"
    df_results.to_csv(out_path, index=False)

    print("=" * 75)
    print("  BASELINE RESULTS")
    print("=" * 75)
    print(df_results.to_string(index=False, float_format="%.4f"))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
