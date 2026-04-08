"""
ECFP fingerprint + 분자 그래프 feature 생성.

Usage:
    python src/features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "data" / "features"


# ── ECFP ──────────────────────────────────────────────────────────────

def smiles_to_ecfp(smi, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(nbits, dtype=np.int8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp, dtype=np.int8)


def generate_ecfp(smiles_list):
    print("[ECFP] Generating Morgan fingerprints (radius=2, 2048 bits)...")
    X = np.array([smiles_to_ecfp(s) for s in smiles_list])
    print(f"  → shape: {X.shape}, dtype: {X.dtype}")
    return X


# ── 분자 그래프 ───────────────────────────────────────────────────────

ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-2, -1, 0, 1, 2],
    "num_hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}


def one_hot(value, allowable_set):
    encoding = [0] * (len(allowable_set) + 1)
    if value in allowable_set:
        encoding[allowable_set.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding


def atom_features(atom):
    features = []
    features += one_hot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
    features += one_hot(atom.GetDegree(), ATOM_FEATURES["degree"])
    features += one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
    features += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"])
    features += one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(1 if atom.IsInRing() else 0)
    return features


def bond_features(bond):
    bt = bond.GetBondType()
    features = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]
    stereo = bond.GetStereo()
    features += one_hot(
        stereo,
        [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ],
    )
    return [float(f) for f in features]


def smiles_to_graph(smi, label):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    # Node features
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    # Edges (bidirectional)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bf)
        edge_attr.append(bf)

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 11), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    y = torch.tensor([label], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def generate_graphs(smiles_list, labels):
    print("[Graph] Converting SMILES to PyG Data objects...")
    graphs = []
    failed = 0
    for smi, lab in zip(smiles_list, labels):
        g = smiles_to_graph(smi, lab)
        if g is None:
            failed += 1
            # placeholder — should not happen after cleaning
            g = Data(
                x=torch.zeros((1, 139), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 11), dtype=torch.float),
                y=torch.tensor([lab], dtype=torch.float),
            )
        graphs.append(g)

    if failed:
        print(f"  *** WARNING: {failed} molecules failed to convert ***")

    # 통계
    n_nodes = [g.num_nodes for g in graphs]
    n_edges = [g.num_edges for g in graphs]
    print(f"  → {len(graphs)} graphs")
    print(f"  → node feature dim: {graphs[0].x.shape[1]}")
    print(f"  → edge feature dim: {graphs[0].edge_attr.shape[1] if graphs[0].edge_attr.shape[0] > 0 else 'N/A'}")
    print(f"  → avg nodes: {np.mean(n_nodes):.1f}, avg edges: {np.mean(n_edges):.1f}")
    return graphs


# ── Main ──────────────────────────────────────────────────────────────

def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "egfr_cleaned.csv")
    smiles = df["clean_smiles"].tolist()
    labels = df["label"].tolist()
    print(f"Loaded {len(df)} molecules\n")

    # ECFP
    X_ecfp = generate_ecfp(smiles)
    ecfp_path = FEATURES_DIR / "ecfp_2048.npy"
    np.save(ecfp_path, X_ecfp)
    print(f"  → Saved {ecfp_path}\n")

    # Graphs
    graphs = generate_graphs(smiles, labels)
    graphs_path = FEATURES_DIR / "graphs.pt"
    torch.save(graphs, graphs_path)
    print(f"  → Saved {graphs_path}")


if __name__ == "__main__":
    main()
