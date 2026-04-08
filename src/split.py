"""
Random split + Scaffold split 생성.

Usage:
    python src/split.py
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"

SEED = 42
FRAC_TRAIN = 0.8
FRAC_VALID = 0.1
FRAC_TEST = 0.1


# ── Random Split ──────────────────────────────────────────────────────

def random_split(df):
    train_idx, temp_idx = train_test_split(
        df.index.values,
        train_size=FRAC_TRAIN,
        stratify=df["label"],
        random_state=SEED,
    )
    valid_idx, test_idx = train_test_split(
        temp_idx,
        train_size=FRAC_VALID / (1 - FRAC_TRAIN),
        stratify=df.loc[temp_idx, "label"],
        random_state=SEED,
    )
    return train_idx, valid_idx, test_idx


# ── Scaffold Split ────────────────────────────────────────────────────

def scaffold_split(df):
    np.random.seed(SEED)

    scaffold_to_indices = defaultdict(list)
    for idx, row in df.iterrows():
        scaffold_to_indices[row["scaffold"]].append(idx)

    # 큰 그룹부터 배치
    scaffold_sets = list(scaffold_to_indices.values())
    scaffold_sets.sort(key=len, reverse=True)

    n_total = len(df)
    n_train = int(n_total * FRAC_TRAIN)
    n_valid = int(n_total * FRAC_VALID)

    train_idx, valid_idx, test_idx = [], [], []

    for group in scaffold_sets:
        if len(train_idx) + len(group) <= n_train:
            train_idx.extend(group)
        elif len(valid_idx) + len(group) <= n_valid:
            valid_idx.extend(group)
        else:
            test_idx.extend(group)

    return np.array(train_idx), np.array(valid_idx), np.array(test_idx)


# ── 검증 ──────────────────────────────────────────────────────────────

def verify_split(df, train_idx, valid_idx, test_idx, split_name):
    print(f"\n{'=' * 50}")
    print(f"  Split: {split_name}")
    print(f"{'=' * 50}")

    total = len(train_idx) + len(valid_idx) + len(test_idx)
    for name, idx in [("Train", train_idx), ("Valid", valid_idx), ("Test", test_idx)]:
        subset = df.loc[idx]
        print(f"  {name:5s}: {len(idx):5d} ({len(idx)/total*100:5.1f}%)  "
              f"active {subset['label'].mean()*100:5.1f}%  "
              f"scaffolds {subset['scaffold'].nunique():4d}")

    train_scaffolds = set(df.loc[train_idx, "scaffold"])
    test_scaffolds = set(df.loc[test_idx, "scaffold"])
    overlap = train_scaffolds & test_scaffolds
    print(f"  Train/Test scaffold overlap: {len(overlap)}")

    if split_name == "scaffold" and len(overlap) > 0:
        print("  *** WARNING: scaffold split has overlapping scaffolds! ***")
    elif split_name == "scaffold":
        print("  *** OK: zero overlap ***")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "egfr_cleaned.csv")
    print(f"Loaded {len(df)} molecules")

    # Random split
    r_train, r_valid, r_test = random_split(df)
    np.save(SPLITS_DIR / "random_train.npy", r_train)
    np.save(SPLITS_DIR / "random_valid.npy", r_valid)
    np.save(SPLITS_DIR / "random_test.npy", r_test)
    verify_split(df, r_train, r_valid, r_test, "random")

    # Scaffold split
    s_train, s_valid, s_test = scaffold_split(df)
    np.save(SPLITS_DIR / "scaffold_train.npy", s_train)
    np.save(SPLITS_DIR / "scaffold_valid.npy", s_valid)
    np.save(SPLITS_DIR / "scaffold_test.npy", s_test)
    verify_split(df, s_train, s_valid, s_test, "scaffold")

    print(f"\nSaved splits → {SPLITS_DIR}")


if __name__ == "__main__":
    main()
