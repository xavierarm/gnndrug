"""
GNN 모델: GCN / GIN / D-MPNN.
3 models x 2 splits = 6 experiments.

Usage:
    OMP_NUM_THREADS=1 python src/gnn.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    NNConv,
    global_add_pool,
    global_mean_pool,
)

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
HIDDEN_DIM = 128
DROPOUT = 0.2


# ── 평가 메트릭 (baseline.py와 동일) ─────────────────────────────────

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

def load_graph_data(split_name, graphs):
    train_idx = np.load(SPLITS_DIR / f"{split_name}_train.npy")
    valid_idx = np.load(SPLITS_DIR / f"{split_name}_valid.npy")
    test_idx = np.load(SPLITS_DIR / f"{split_name}_test.npy")

    train_data = [graphs[i] for i in train_idx]
    valid_data = [graphs[i] for i in valid_idx]
    test_data = [graphs[i] for i in test_idx]

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    return train_loader, valid_loader, test_loader


# ── GCN ───────────────────────────────────────────────────────────────

class GCNModel(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=HIDDEN_DIM, num_layers=4,
                 dropout=DROPOUT):
        super().__init__()
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
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
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_encoder(x)
        for conv, bn in zip(self.convs, self.bns):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # residual
        x_graph = torch.cat([global_mean_pool(x, batch),
                             global_add_pool(x, batch)], dim=-1)
        return self.head(x_graph).squeeze(-1)


# ── GIN ───────────────────────────────────────────────────────────────

class GINModel(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=HIDDEN_DIM, num_layers=5,
                 dropout=DROPOUT):
        super().__init__()
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.num_layers = num_layers
        # Jumping Knowledge: all layer outputs concatenated
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * (num_layers + 1), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_encoder(x)
        layer_outputs = [global_mean_pool(x, batch)]
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(global_mean_pool(x, batch))
        x_graph = torch.cat(layer_outputs, dim=-1)
        return self.head(x_graph).squeeze(-1)


# ── D-MPNN ────────────────────────────────────────────────────────────

class DMPNNModel(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=HIDDEN_DIM,
                 num_layers=4, dropout=DROPOUT):
        super().__init__()
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            # 2-layer MLP: edge_dim → 128 → hidden² (much lighter than direct)
            edge_nn = nn.Sequential(
                nn.Linear(edge_feat_dim, 128),
                nn.ReLU(),
                nn.Linear(128, hidden_dim * hidden_dim),
            )
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr="mean"))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        x = self.node_encoder(x)
        for conv, bn in zip(self.convs, self.bns):
            x_new = conv(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # residual
        x_graph = global_mean_pool(x, batch)
        return self.head(x_graph).squeeze(-1)


# ── 학습 루프 ─────────────────────────────────────────────────────────

def compute_pos_weight(loader):
    labels = []
    for batch in loader:
        labels.extend(batch.y.tolist())
    labels = np.array(labels)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return n_neg / max(n_pos, 1)


@torch.no_grad()
def evaluate_gnn(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        probs = torch.sigmoid(out)
        all_preds.extend(probs.cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())
    auc = roc_auc_score(all_labels, all_preds)
    return auc, all_preds, all_labels


def train_gnn(model, train_loader, valid_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6,
    )

    pw = compute_pos_weight(train_loader)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], device=device)
    )

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
            loss = criterion(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n_graphs += batch.num_graphs

        valid_auc, _, _ = evaluate_gnn(model, valid_loader, device)
        scheduler.step(valid_auc)

        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= PATIENCE:
            break

        if epoch % 20 == 0:
            avg_loss = total_loss / n_graphs
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"    ep {epoch:3d} | loss {avg_loss:.4f} | "
                  f"val_auc {valid_auc:.4f} | lr {lr_now:.1e}")

    print(f"    → best epoch {best_epoch}, val_auc {best_auc:.4f}")
    model.load_state_dict(best_state)
    return model


# ── Main ──────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading graphs...", end=" ", flush=True)
    graphs = torch.load(FEATURES_DIR / "graphs.pt", weights_only=False)
    node_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    print(f"{len(graphs)} graphs, node_dim={node_dim}, edge_dim={edge_dim}\n")

    model_configs = {
        "GCN": lambda: GCNModel(node_dim),
        "GIN": lambda: GINModel(node_dim),
        "D-MPNN": lambda: DMPNNModel(node_dim, edge_dim),
    }

    all_results = []

    for split_name in ["random", "scaffold"]:
        train_loader, valid_loader, test_loader = load_graph_data(split_name, graphs)
        print(f"{'=' * 55}")
        print(f"  Split: {split_name}")
        print(f"{'=' * 55}")

        for model_name, model_fn in model_configs.items():
            print(f"  [{model_name}] training...")
            torch.manual_seed(SEED)
            model = model_fn().to(device)

            model = train_gnn(model, train_loader, valid_loader, device)

            # test evaluation
            _, test_preds, test_labels = evaluate_gnn(model, test_loader, device)
            metrics = evaluate(np.array(test_labels), np.array(test_preds))
            metrics.update({"Model": model_name, "Split": split_name})
            all_results.append(metrics)
            print(f"    TEST  ROC-AUC={metrics['ROC-AUC']:.4f}  "
                  f"PR-AUC={metrics['PR-AUC']:.4f}\n")

    # 결과 정리
    df = pd.DataFrame(all_results)
    col_order = ["Model", "Split", "ROC-AUC", "PR-AUC", "EF1%", "EF5%", "P@50", "P@100"]
    df = df[col_order]

    out_path = RESULTS_DIR / "gnn_performance.csv"
    df.to_csv(out_path, index=False)

    print("=" * 75)
    print("  GNN RESULTS")
    print("=" * 75)
    print(df.to_string(index=False, float_format="%.4f"))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
