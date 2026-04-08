"""
Shared utilities: evaluation metrics, GNN training loop, model definitions.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "data" / "features"
SPLITS_DIR = ROOT / "data" / "splits"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


# ── Evaluation Metrics ────────────────────────────────────────────────

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


# ── GNN Utilities ─────────────────────────────────────────────────────

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
            all_preds.extend(out.cpu().tolist())
        else:
            all_preds.extend(torch.sigmoid(out).cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())
    auc = roc_auc_score(all_labels, all_preds)
    return auc, all_preds, all_labels


def train_gnn(model, train_loader, valid_loader, device,
              lr=1e-3, weight_decay=1e-5, epochs=200, patience=20,
              regression=False, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6)

    if regression:
        criterion = nn.MSELoss()
    else:
        pw = compute_pos_weight(train_loader)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))

    best_auc, best_epoch, best_state = 0, 0, None

    for epoch in range(epochs):
        model.train()
        total_loss, n_graphs = 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            target = batch.y_reg.to(device) if regression else batch.y
            loss = criterion(out, target)
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

        if epoch - best_epoch >= patience:
            break

        if verbose and epoch % 20 == 0:
            print(f"    ep {epoch:3d} | loss {total_loss/n_graphs:.4f} | "
                  f"val_auc {valid_auc:.4f} | lr {optimizer.param_groups[0]['lr']:.1e}")

    if verbose:
        print(f"    -> best epoch {best_epoch}, val_auc {best_auc:.4f}")
    model.load_state_dict(best_state)
    return model


# ── Model Definitions ─────────────────────────────────────────────────

class GCN256(nn.Module):
    """GCN with residual connections + mean/sum pooling."""
    def __init__(self, node_dim, hidden_dim=256, num_layers=4, dropout=0.2, out_dim=1):
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
            nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))

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
