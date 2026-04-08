"""
Enhanced analysis: scaffold leakage, regression threshold analysis,
distance table with n, EF practical interpretation.

Usage:
    OMP_NUM_THREADS=1 python src/analysis_v2.py
"""

import sys
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    FEATURES_DIR, SPLITS_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR,
    compute_pos_weight, evaluate_gnn, train_gnn, GCN256,
)

SEED = 42
plt.style.use("seaborn-v0_8-whitegrid")


# ── 1. Scaffold Split Leakage ─────────────────────────────────────────

def quantify_leakage(df, X_ecfp, idx):
    print("[1] Scaffold split leakage quantification...")

    train_fps = []
    for i in idx["train"]:
        mol = Chem.MolFromSmiles(df.iloc[i]["clean_smiles"])
        if mol:
            train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    high_sim_count = 0
    high_sim_pairs = []
    test_max_sims = []

    for ti, i in enumerate(idx["test"]):
        mol = Chem.MolFromSmiles(df.iloc[i]["clean_smiles"])
        if not mol:
            test_max_sims.append(0)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        max_sim = max(sims)
        test_max_sims.append(max_sim)

        if max_sim >= 0.7:
            high_sim_count += 1

    test_max_sims = np.array(test_max_sims)
    n_test = len(idx["test"])

    print(f"  Test molecules with Tanimoto >= 0.7 to any train mol: "
          f"{high_sim_count}/{n_test} ({high_sim_count/n_test*100:.1f}%)")
    print(f"  Test molecules with Tanimoto >= 0.5: "
          f"{(test_max_sims >= 0.5).sum()}/{n_test} ({(test_max_sims >= 0.5).sum()/n_test*100:.1f}%)")
    print(f"  Mean max Tanimoto: {test_max_sims.mean():.3f}")
    print(f"  Median max Tanimoto: {np.median(test_max_sims):.3f}")

    return test_max_sims, high_sim_count


# ── 2. Regression vs Classification Threshold Analysis ────────────────

def regression_threshold_analysis(graphs, df, idx, device, path):
    print("\n[2] Regression vs Classification threshold analysis...")
    np.random.seed(SEED); torch.manual_seed(SEED)
    node_dim = graphs[0].x.shape[1]
    pchembl = df["pchembl_value"].values

    train_loader = DataLoader([graphs[i] for i in idx["train"]], batch_size=64, shuffle=True)
    valid_loader = DataLoader([graphs[i] for i in idx["valid"]], batch_size=64)
    test_loader = DataLoader([graphs[i] for i in idx["test"]], batch_size=64)

    # Classification model
    torch.manual_seed(SEED)
    cls_model = GCN256(node_dim).to(device)
    cls_model = train_gnn(cls_model, train_loader, valid_loader, device, verbose=False)
    cls_model.eval()
    cls_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            cls_preds.extend(torch.sigmoid(cls_model(batch)).cpu().tolist())
    cls_preds = np.array(cls_preds)

    # Regression model — need to add y_reg to graphs
    graphs_reg = []
    for i, g in enumerate(graphs):
        g2 = deepcopy(g)
        g2.y_reg = torch.tensor([pchembl[i]], dtype=torch.float)
        graphs_reg.append(g2)

    train_loader_reg = DataLoader([graphs_reg[i] for i in idx["train"]], batch_size=64, shuffle=True)
    valid_loader_reg = DataLoader([graphs_reg[i] for i in idx["valid"]], batch_size=64)
    test_loader_reg = DataLoader([graphs_reg[i] for i in idx["test"]], batch_size=64)

    torch.manual_seed(SEED)
    reg_model = GCN256(node_dim).to(device)
    reg_model = train_gnn(reg_model, train_loader_reg, valid_loader_reg, device,
                          regression=True, verbose=False)
    reg_model.eval()
    reg_preds = []
    with torch.no_grad():
        for batch in test_loader_reg:
            batch = batch.to(device)
            reg_preds.extend(reg_model(batch).cpu().tolist())
    reg_preds = np.array(reg_preds)

    # True pIC50 values for test set
    true_pchembl = pchembl[idx["test"]]
    y_test = df["label"].values[idx["test"]]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Classification output distribution
    ax = axes[0]
    active_mask = y_test == 1
    ax.hist(cls_preds[active_mask], bins=50, alpha=0.6, label="Active", color="#2196F3", density=True)
    ax.hist(cls_preds[~active_mask], bins=50, alpha=0.6, label="Inactive", color="#FF5722", density=True)
    ax.axvline(x=0.5, color="black", linestyle="--", label="Threshold=0.5")
    ax.set_xlabel("Predicted probability (sigmoid)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Classification: Prediction Distribution", fontsize=13)
    ax.legend(fontsize=11)

    # Right: Regression output distribution (focus on threshold region)
    ax = axes[1]
    ax.hist(reg_preds[active_mask], bins=50, alpha=0.6, label="Active", color="#2196F3", density=True)
    ax.hist(reg_preds[~active_mask], bins=50, alpha=0.6, label="Inactive", color="#FF5722", density=True)
    ax.axvline(x=7.0, color="black", linestyle="--", label="Threshold=7.0")
    ax.set_xlabel("Predicted pIC50", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Regression: Prediction Distribution", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(4, 10)

    fig.suptitle("Why Regression Hurts: Decision Boundary Analysis (Scaffold Split)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # Overlap quantification
    cls_near_boundary = ((cls_preds >= 0.3) & (cls_preds <= 0.7)).sum()
    reg_near_boundary = ((reg_preds >= 6.0) & (reg_preds <= 8.0)).sum()
    print(f"  Classification: {cls_near_boundary}/{len(cls_preds)} "
          f"({cls_near_boundary/len(cls_preds)*100:.1f}%) near boundary [0.3-0.7]")
    print(f"  Regression:     {reg_near_boundary}/{len(reg_preds)} "
          f"({reg_near_boundary/len(reg_preds)*100:.1f}%) near boundary [6.0-8.0]")


# ── 3. Distance vs Performance (enhanced table) ──────────────────────

def distance_table(test_max_sims, y_test, rf_preds, gcn_preds):
    print("\n[3] Distance vs Performance (detailed table)...")
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]

    print(f"\n  {'Tanimoto':<15} {'n':>5} {'Active%':>8} {'ECFP+RF':>10} {'GCN-256':>10} {'Delta':>8}")
    print(f"  {'-'*58}")

    for low, high in bins:
        mask = (test_max_sims >= low) & (test_max_sims < high)
        n = mask.sum()
        act_pct = y_test[mask].mean() * 100 if n > 0 else 0

        if n > 20 and len(set(y_test[mask])) > 1:
            rf_auc = roc_auc_score(y_test[mask], rf_preds[mask])
            gc_auc = roc_auc_score(y_test[mask], gcn_preds[mask])
            delta = gc_auc - rf_auc
            print(f"  [{low:.1f}, {high:.1f}){'':<7} {n:>5} {act_pct:>7.1f}% "
                  f"{rf_auc:>10.4f} {gc_auc:>10.4f} {delta:>+8.4f}")
        else:
            print(f"  [{low:.1f}, {high:.1f}){'':<7} {n:>5} {act_pct:>7.1f}%     (insufficient data)")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    df = pd.read_csv(PROCESSED_DIR / "egfr_cleaned.csv")
    X_ecfp = np.load(FEATURES_DIR / "ecfp_2048.npy")
    graphs = torch.load(FEATURES_DIR / "graphs.pt", weights_only=False)
    idx = {s: np.load(SPLITS_DIR / f"scaffold_{s}.npy")
           for s in ["train", "valid", "test"]}
    y_test = df["label"].values[idx["test"]]

    # 1. Leakage
    test_max_sims, high_sim_count = quantify_leakage(df, X_ecfp, idx)

    # 2. Regression threshold
    regression_threshold_analysis(graphs, df, idx, device,
                                  FIGURES_DIR / "regression_threshold_analysis.png")

    # 3. Distance table (reuse RF + GCN predictions from analysis.py)
    # Quick re-train for predictions
    from sklearn.ensemble import RandomForestClassifier
    np.random.seed(SEED)
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                class_weight="balanced", n_jobs=-1, random_state=SEED)
    rf.fit(X_ecfp[idx["train"]], df["label"].values[idx["train"]])
    rf_preds = rf.predict_proba(X_ecfp[idx["test"]])[:, 1]

    torch.manual_seed(SEED)
    node_dim = graphs[0].x.shape[1]
    train_loader = DataLoader([graphs[i] for i in idx["train"]], batch_size=64, shuffle=True)
    valid_loader = DataLoader([graphs[i] for i in idx["valid"]], batch_size=64)
    test_loader = DataLoader([graphs[i] for i in idx["test"]], batch_size=64)
    gcn = GCN256(node_dim).to(device)
    gcn = train_gnn(gcn, train_loader, valid_loader, device, verbose=False)
    _, gcn_preds, _ = evaluate_gnn(gcn, test_loader, device)
    gcn_preds = np.array(gcn_preds)

    distance_table(test_max_sims, y_test, rf_preds, gcn_preds)

    # 4. EF practical interpretation
    n_test = len(y_test)
    active_rate = y_test.mean()
    ef1 = 2.57  # ECFP+RF EF1%
    print(f"\n[4] EF Practical Interpretation:")
    print(f"  Test set: {n_test} molecules, {active_rate*100:.1f}% active")
    print(f"  ECFP+RF EF1% = {ef1:.2f}")
    print(f"  -> In a 10,000-compound library (~{active_rate*100:.0f}% active):")
    n_screen = 100
    expected_random = active_rate * n_screen
    expected_model = active_rate * ef1 * n_screen
    print(f"     Screen top {n_screen}: expect ~{expected_model:.0f} actives "
          f"(vs ~{expected_random:.0f} by random)")
    print(f"     That's {ef1:.1f}x more efficient than random selection")

    print("\n[Done]")


if __name__ == "__main__":
    main()
