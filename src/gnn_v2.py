"""
GNN Ablation Study: hidden=256 복원 + regression + virtual node + AttentiveFP.
5 model variants x 2 splits = 10 experiments.

Usage:
    python3 -u src/gnn_v2.py
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    global_add_pool,
    global_mean_pool,
)
from torch_geometric.nn.models import AttentiveFP as PyGAttentiveFP

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "data" / "features"
SPLITS_DIR = ROOT / "data" / "splits"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

SEED = 42
BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 20
LR = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 256
DROPOUT = 0.2
PCHEMBL_THRESHOLD = 7.0


# ── 평가 ──────────────────────────────────────────────────────────────

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


# ── 데이터 ────────────────────────────────────────────────────────────

def load_graphs_with_pchembl(graphs, pchembl_values):
    """그래프 리스트에 pchembl_value를 y_reg로 추가."""
    out = []
    for i, g in enumerate(graphs):
        g2 = deepcopy(g)
        g2.y_reg = torch.tensor([pchembl_values[i]], dtype=torch.float)
        out.append(g2)
    return out


def add_virtual_node(graph):
    """그래프에 virtual node 추가. 모든 실제 노드와 양방향 연결."""
    g = deepcopy(graph)
    n = g.x.shape[0]
    node_dim = g.x.shape[1]
    edge_dim = g.edge_attr.shape[1] if g.edge_attr.shape[0] > 0 else 11

    # virtual node feature = zero vector
    vn_feat = torch.zeros(1, node_dim)
    g.x = torch.cat([g.x, vn_feat], dim=0)

    # virtual node ↔ all real nodes (bidirectional)
    vn_edges_src = list(range(n)) + [n] * n
    vn_edges_dst = [n] * n + list(range(n))
    vn_edge_index = torch.tensor([vn_edges_src, vn_edges_dst], dtype=torch.long)
    g.edge_index = torch.cat([g.edge_index, vn_edge_index], dim=1)

    # virtual edges get zero edge features
    vn_edge_attr = torch.zeros(2 * n, edge_dim)
    g.edge_attr = torch.cat([g.edge_attr, vn_edge_attr], dim=0)

    return g


def make_loaders(graphs, split_name, batch_size=BATCH_SIZE):
    train_idx = np.load(SPLITS_DIR / f"{split_name}_train.npy")
    valid_idx = np.load(SPLITS_DIR / f"{split_name}_valid.npy")
    test_idx = np.load(SPLITS_DIR / f"{split_name}_test.npy")
    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader([graphs[i] for i in valid_idx], batch_size=batch_size)
    test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=batch_size)
    return train_loader, valid_loader, test_loader


# ── 모델 ──────────────────────────────────────────────────────────────

class GCN256(nn.Module):
    """GCN with hidden_dim=256, residual + mean/sum pooling."""
    def __init__(self, node_dim, hidden_dim=HIDDEN_DIM, num_layers=4,
                 dropout=DROPOUT, out_dim=1):
        super().__init__()
        self.encoder = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
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


class AttentiveFPModel(nn.Module):
    """PyG AttentiveFP wrapper."""
    def __init__(self, node_dim, edge_dim, hidden_dim=HIDDEN_DIM,
                 num_layers=4, num_timesteps=2, dropout=DROPOUT, out_dim=1):
        super().__init__()
        self.model = PyGAttentiveFP(
            in_channels=node_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

    def forward(self, data):
        out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
        return out.squeeze(-1)


# ── 학습 ──────────────────────────────────────────────────────────────

def compute_pos_weight(loader):
    labels = []
    for batch in loader:
        labels.extend(batch.y.tolist())
    labels = np.array(labels)
    n_pos = labels.sum()
    return (len(labels) - n_pos) / max(n_pos, 1)


@torch.no_grad()
def evaluate_gnn(model, loader, device, regression=False):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        if regression:
            # pIC50 예측 → threshold로 probability-like score 생성
            # higher pIC50 = more active → 바로 score로 사용
            all_preds.extend(out.cpu().tolist())
        else:
            all_preds.extend(torch.sigmoid(out).cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())
    auc = roc_auc_score(all_labels, all_preds)
    return auc, all_preds, all_labels


def train_gnn(model, train_loader, valid_loader, device, regression=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6,
    )

    if regression:
        criterion = nn.MSELoss()
    else:
        pw = compute_pos_weight(train_loader)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))

    best_auc = 0
    best_epoch = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_graphs = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            if regression:
                loss = criterion(out, batch.y_reg.to(device))
            else:
                loss = criterion(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n_graphs += batch.num_graphs

        valid_auc, _, _ = evaluate_gnn(model, valid_loader, device, regression)
        scheduler.step(valid_auc)

        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= PATIENCE:
            break

        if epoch % 20 == 0:
            avg_loss = total_loss / n_graphs
            print(f"    ep {epoch:3d} | loss {avg_loss:.4f} | "
                  f"val_auc {valid_auc:.4f} | lr {optimizer.param_groups[0]['lr']:.1e}")

    print(f"    → best epoch {best_epoch}, val_auc {best_auc:.4f}")
    model.load_state_dict(best_state)
    return model


# ── 실험 정의 ─────────────────────────────────────────────────────────

def run_experiment(name, model_fn, graphs, device, regression=False):
    """하나의 모델을 random + scaffold split에서 학습/평가."""
    results = []
    for split_name in ["random", "scaffold"]:
        print(f"\n  [{name}] split={split_name}")
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        train_loader, valid_loader, test_loader = make_loaders(graphs, split_name)
        model = model_fn().to(device)
        model = train_gnn(model, train_loader, valid_loader, device, regression)

        _, test_preds, test_labels = evaluate_gnn(model, test_loader, device, regression)
        metrics = evaluate(np.array(test_labels), np.array(test_preds))
        metrics.update({"Model": name, "Split": split_name})
        results.append(metrics)
        print(f"    TEST  ROC-AUC={metrics['ROC-AUC']:.4f}  PR-AUC={metrics['PR-AUC']:.4f}")

    return results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load base graphs + pchembl values
    print("Loading graphs...", end=" ", flush=True)
    graphs_base = torch.load(FEATURES_DIR / "graphs.pt", weights_only=False)
    node_dim = graphs_base[0].x.shape[1]
    edge_dim = graphs_base[0].edge_attr.shape[1]
    print(f"{len(graphs_base)} graphs, node_dim={node_dim}, edge_dim={edge_dim}")

    df = pd.read_csv(PROCESSED_DIR / "egfr_cleaned.csv")
    pchembl = df["pchembl_value"].values

    # Prepare graph variants
    print("Preparing graph variants...")
    graphs_cls = load_graphs_with_pchembl(graphs_base, pchembl)  # classification
    graphs_vn = [add_virtual_node(g) for g in graphs_cls]  # + virtual node
    print("  Done.\n")

    all_results = []

    # ── Exp 1: GCN-256 (classification) ──
    all_results += run_experiment(
        "GCN-256",
        lambda: GCN256(node_dim),
        graphs_cls, device, regression=False,
    )

    # ── Exp 2: GCN-256-reg (regression) ──
    all_results += run_experiment(
        "GCN-256-reg",
        lambda: GCN256(node_dim),
        graphs_cls, device, regression=True,
    )

    # ── Exp 3: GCN-256-vn (virtual node) ──
    all_results += run_experiment(
        "GCN-256-vn",
        lambda: GCN256(node_dim),
        graphs_vn, device, regression=False,
    )

    # ── Exp 4: AttentiveFP (classification) ──
    all_results += run_experiment(
        "AttentiveFP",
        lambda: AttentiveFPModel(node_dim, edge_dim),
        graphs_cls, device, regression=False,
    )

    # ── Exp 5: AttentiveFP + reg + vnode (best combo) ──
    all_results += run_experiment(
        "AttFP-reg-vn",
        lambda: AttentiveFPModel(node_dim, edge_dim),
        graphs_vn, device, regression=True,
    )

    # 결과 정리
    res_df = pd.DataFrame(all_results)
    col_order = ["Model", "Split", "ROC-AUC", "PR-AUC", "EF1%", "EF5%", "P@50", "P@100"]
    res_df = res_df[col_order]

    out_path = RESULTS_DIR / "gnn_v2_performance.csv"
    res_df.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("  GNN v2 ABLATION RESULTS")
    print("=" * 80)
    print(res_df.to_string(index=False, float_format="%.4f"))

    # 기존 결과와 비교
    print("\n" + "=" * 80)
    print("  SCAFFOLD SPLIT COMPARISON (with previous results)")
    print("=" * 80)
    scaffold = res_df[res_df["Split"] == "scaffold"][["Model", "ROC-AUC", "PR-AUC", "EF1%"]].copy()
    # 기존 결과 추가
    prev = pd.DataFrame([
        {"Model": "ECFP+RF (best baseline)", "ROC-AUC": 0.8200, "PR-AUC": 0.7726, "EF1%": 2.5658},
        {"Model": "GCN-128 (prev)", "ROC-AUC": 0.7878, "PR-AUC": 0.7291, "EF1%": 2.3092},
    ])
    comparison = pd.concat([prev, scaffold], ignore_index=True)
    print(comparison.to_string(index=False, float_format="%.4f"))

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
