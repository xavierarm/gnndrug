"""
Phase 4: Screening Analysis + Phase 5: Error & Uncertainty Analysis.
Scaffold split 기준, ECFP+RF vs GCN-256 비교.

Usage:
    OMP_NUM_THREADS=1 python src/analysis.py
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "data" / "features"
SPLITS_DIR = ROOT / "data" / "splits"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

SEED = 42
plt.style.use("seaborn-v0_8-whitegrid")


# ── GCN-256 모델 (gnn_v2.py에서 가져옴) ──────────────────────────────

class GCN256(nn.Module):
    def __init__(self, node_dim, hidden_dim=256, num_layers=4, dropout=0.2):
        super().__init__()
        self.encoder = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.encoder(x)
        for conv, bn in zip(self.convs, self.bns):
            x_new = bn(F.relu(conv(x, edge_index)))
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new
        x_graph = torch.cat([global_mean_pool(x, batch),
                             global_add_pool(x, batch)], dim=-1)
        return self.head(x_graph).squeeze(-1)


# ── 데이터 로딩 ──────────────────────────────────────────────────────

def load_all():
    df = pd.read_csv(PROCESSED_DIR / "egfr_cleaned.csv")
    X_ecfp = np.load(FEATURES_DIR / "ecfp_2048.npy")
    graphs = torch.load(FEATURES_DIR / "graphs.pt", weights_only=False)

    idx = {s: np.load(SPLITS_DIR / f"scaffold_{s}.npy")
           for s in ["train", "valid", "test"]}
    return df, X_ecfp, graphs, idx


# ── 모델 학습 (scaffold split) ────────────────────────────────────────

def train_rf(X, y, idx):
    model = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, class_weight="balanced",
        n_jobs=-1, random_state=SEED)
    model.fit(X[idx["train"]], y[idx["train"]])
    return model.predict_proba(X[idx["test"]])[:, 1]


def train_gcn(graphs, idx, device):
    np.random.seed(SEED); torch.manual_seed(SEED)
    node_dim = graphs[0].x.shape[1]

    train_loader = DataLoader([graphs[i] for i in idx["train"]], batch_size=64, shuffle=True)
    valid_loader = DataLoader([graphs[i] for i in idx["valid"]], batch_size=64)
    test_loader = DataLoader([graphs[i] for i in idx["test"]], batch_size=64)

    # pos_weight
    labels = [graphs[i].y.item() for i in idx["train"]]
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    model = GCN256(node_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6)

    best_auc, best_state = 0, None
    patience_counter = 0
    for epoch in range(200):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        preds, labels_v = [], []
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                preds.extend(torch.sigmoid(model(batch)).cpu().tolist())
                labels_v.extend(batch.y.cpu().tolist())
        auc = roc_auc_score(labels_v, preds)
        scheduler.step(auc)

        if auc > best_auc:
            best_auc = auc; patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 20: break

    model.load_state_dict(best_state)

    # test predictions
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            test_preds.extend(torch.sigmoid(model(batch)).cpu().tolist())

    return np.array(test_preds), model, test_loader


# ── Phase 4: Enrichment Curves ────────────────────────────────────────

def plot_enrichment(y_true, preds_dict, path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    percentages = np.arange(1, 101)

    for name, preds in preds_dict.items():
        efs = []
        for p in percentages:
            n_top = max(1, int(len(y_true) * p / 100))
            top_idx = np.argsort(preds)[::-1][:n_top]
            hit_rate = y_true[top_idx].sum() / n_top
            efs.append(hit_rate)
        ax.plot(percentages, efs, label=name, linewidth=2)

    # random baseline
    baseline_rate = y_true.mean()
    ax.axhline(y=baseline_rate, color="gray", linestyle="--", label=f"Random ({baseline_rate:.2f})")
    ax.set_xlabel("Top x% of ranked compounds", fontsize=13)
    ax.set_ylabel("Active fraction in top x%", fontsize=13)
    ax.set_title("Enrichment Curves (Scaffold Split)", fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xlim(0, 50)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Phase 5a: Error Analysis ──────────────────────────────────────────

def error_analysis(df_test, y_true, preds, model_name):
    y_pred = (preds >= 0.5).astype(int)
    fp = np.where((y_pred == 1) & (y_true == 0))[0]
    fn = np.where((y_pred == 0) & (y_true == 1))[0]
    tp = np.where((y_pred == 1) & (y_true == 1))[0]
    tn = np.where((y_pred == 0) & (y_true == 0))[0]

    lines = [f"## Error Analysis: {model_name} (Scaffold Split)\n"]
    lines.append(f"- TP: {len(tp)}, TN: {len(tn)}, FP: {len(fp)}, FN: {len(fn)}")
    lines.append(f"- Accuracy: {(len(tp)+len(tn))/len(y_true):.3f}\n")

    # Property comparison
    props = {"MolWt": Descriptors.MolWt, "MolLogP": Descriptors.MolLogP,
             "TPSA": Descriptors.TPSA, "NumAromaticRings": Descriptors.NumAromaticRings}

    lines.append("### Molecular Properties: Error vs Correct\n")
    lines.append(f"{'Property':<20} {'Error mean':>12} {'Correct mean':>14} {'Diff%':>8}")
    lines.append("-" * 58)

    error_idx = np.concatenate([fp, fn])
    correct_idx = np.concatenate([tp, tn])

    for name, func in props.items():
        err_vals = [func(Chem.MolFromSmiles(df_test.iloc[i]["clean_smiles"]))
                    for i in error_idx if Chem.MolFromSmiles(df_test.iloc[i]["clean_smiles"])]
        cor_vals = [func(Chem.MolFromSmiles(df_test.iloc[i]["clean_smiles"]))
                    for i in correct_idx if Chem.MolFromSmiles(df_test.iloc[i]["clean_smiles"])]
        if err_vals and cor_vals:
            em, cm = np.mean(err_vals), np.mean(cor_vals)
            diff = (em - cm) / cm * 100 if cm != 0 else 0
            lines.append(f"{name:<20} {em:>12.2f} {cm:>14.2f} {diff:>+7.1f}%")

    return "\n".join(lines)


# ── Phase 5a: Distance vs Performance ─────────────────────────────────

def distance_vs_performance(df, X_ecfp, y_test, preds_dict, idx, path):
    # Compute max Tanimoto for each test molecule
    train_fps = []
    for i in idx["train"]:
        mol = Chem.MolFromSmiles(df.iloc[i]["clean_smiles"])
        if mol:
            train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    max_sims = []
    for i in idx["test"]:
        mol = Chem.MolFromSmiles(df.iloc[i]["clean_smiles"])
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            max_sims.append(max(sims))
        else:
            max_sims.append(0)

    max_sims = np.array(max_sims)
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]

    fig, ax = plt.subplots(figsize=(8, 6))
    x_positions = np.arange(len(bins))
    width = 0.35

    for offset, (name, preds) in enumerate(preds_dict.items()):
        aucs = []
        for low, high in bins:
            mask = (max_sims >= low) & (max_sims < high)
            n = mask.sum()
            if n > 20 and len(set(y_test[mask])) > 1:
                aucs.append(roc_auc_score(y_test[mask], preds[mask]))
            else:
                aucs.append(0)
        ax.bar(x_positions + offset * width - width/2, aucs, width, label=name)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"[{l:.1f}, {h:.1f})" for l, h in bins])
    ax.set_xlabel("Max Tanimoto similarity to training set", fontsize=13)
    ax.set_ylabel("ROC-AUC", fontsize=13)
    ax.set_title("Performance Degradation by Distance (Scaffold Split)", fontsize=14)
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    return max_sims


# ── Phase 5b: MC Dropout Uncertainty ──────────────────────────────────

def mc_dropout_analysis(model, test_loader, y_test, device, path, n_forward=30):
    model.train()  # keep dropout active
    all_runs = []
    for _ in range(n_forward):
        preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                preds.extend(torch.sigmoid(model(batch)).cpu().tolist())
        all_runs.append(preds)

    all_runs = np.array(all_runs)  # (30, n_test)
    mean_pred = all_runs.mean(axis=0)
    std_pred = all_runs.std(axis=0)

    # Confidence-filtered performance
    fig, ax = plt.subplots(figsize=(8, 6))
    percentiles = [100, 90, 75, 50, 25]
    aucs = []
    ns = []
    for pct in percentiles:
        threshold = np.percentile(std_pred, pct)
        mask = std_pred <= threshold
        n = mask.sum()
        if n > 20 and len(set(y_test[mask])) > 1:
            auc = roc_auc_score(y_test[mask], mean_pred[mask])
        else:
            auc = 0
        aucs.append(auc)
        ns.append(n)

    ax.plot(percentiles, aucs, "o-", linewidth=2, markersize=8, color="#2196F3")
    for i, (p, a, n) in enumerate(zip(percentiles, aucs, ns)):
        ax.annotate(f"n={n}", (p, a), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)
    ax.set_xlabel("Top x% most confident predictions", fontsize=13)
    ax.set_ylabel("ROC-AUC", fontsize=13)
    ax.set_title("GCN-256: Confidence vs Performance (MC Dropout)", fontsize=14)
    ax.set_xlim(110, 15)
    ax.set_ylim(0.6, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    return mean_pred, std_pred


# ── Main ──────────────────────────────────────────────────────────────

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    print(f"Device: {device}\n")

    df, X_ecfp, graphs, idx = load_all()
    y = df["label"].values
    y_test = y[idx["test"]]
    df_test = df.iloc[idx["test"]].reset_index(drop=True)

    # ── Train models & get predictions ──
    print("[1] Training ECFP+RF (scaffold split)...")
    rf_preds = train_rf(X_ecfp, y, idx)
    print(f"    AUC = {roc_auc_score(y_test, rf_preds):.4f}")

    print("[2] Training GCN-256 (scaffold split)...")
    gcn_preds, gcn_model, test_loader = train_gcn(graphs, idx, device)
    print(f"    AUC = {roc_auc_score(y_test, gcn_preds):.4f}")

    preds_dict = {"ECFP+RF": rf_preds, "GCN-256": gcn_preds}

    # ── Phase 4: Enrichment ──
    print("\n[Phase 4] Enrichment curves...")
    plot_enrichment(y_test, preds_dict, FIGURES_DIR / "enrichment_curves.png")

    # Top-50 analysis
    print("  Top-50 hits analysis:")
    for name, preds in preds_dict.items():
        top50 = np.argsort(preds)[::-1][:50]
        n_active = y_test[top50].sum()
        n_scaffolds = df_test.iloc[top50]["scaffold"].nunique()
        print(f"    {name}: {n_active}/50 active, {n_scaffolds} unique scaffolds")

    # ── Phase 5a: Error Analysis ──
    print("\n[Phase 5a] Error analysis...")
    report = error_analysis(df_test, y_test, rf_preds, "ECFP+RF")
    report += "\n\n---\n\n"
    report += error_analysis(df_test, y_test, gcn_preds, "GCN-256")

    # Distance vs performance
    print("[Phase 5a] Distance vs performance...")
    max_sims = distance_vs_performance(
        df, X_ecfp, y_test, preds_dict, idx,
        FIGURES_DIR / "distance_vs_performance.png")

    # Add distance analysis to report
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
    report += "\n\n---\n\n## Distance to Training Set vs Performance\n"
    report += f"\n{'Tanimoto Range':<18} {'n':>5} {'ECFP+RF AUC':>13} {'GCN-256 AUC':>13}"
    report += "\n" + "-" * 52
    for low, high in bins:
        mask = (max_sims >= low) & (max_sims < high)
        n = mask.sum()
        rf_a = roc_auc_score(y_test[mask], rf_preds[mask]) if n > 20 and len(set(y_test[mask])) > 1 else 0
        gc_a = roc_auc_score(y_test[mask], gcn_preds[mask]) if n > 20 and len(set(y_test[mask])) > 1 else 0
        report += f"\n[{low:.1f}, {high:.1f}){'':<10} {n:>5} {rf_a:>13.4f} {gc_a:>13.4f}"

    # ── Phase 5b: Uncertainty ──
    print("\n[Phase 5b] MC Dropout uncertainty analysis...")
    mean_pred, std_pred = mc_dropout_analysis(
        gcn_model, test_loader, y_test, device,
        FIGURES_DIR / "confidence_vs_performance.png")

    # Scaffold novelty vs uncertainty
    scaffold_freq = df_test["scaffold"].map(df_test["scaffold"].value_counts()).values
    corr = np.corrcoef(scaffold_freq, std_pred)[0, 1]
    report += f"\n\n---\n\n## Uncertainty Analysis (GCN-256 MC Dropout)\n"
    report += f"\n- Scaffold frequency vs uncertainty correlation: {corr:.3f}"
    report += f"\n  (Negative = rare scaffolds have higher uncertainty = expected)"

    # Save report
    report_path = RESULTS_DIR / "error_analysis.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n[Done] Report saved → {report_path}")
    print(f"[Done] Figures saved → {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
