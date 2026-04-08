"""
EGFR (CHEMBL203) IC50 데이터 수집 + 정제 파이프라인.

Usage:
    python src/data_pipeline.py
"""

import os
import warnings
from pathlib import Path

import json
import time

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.SaltRemover import SaltRemover

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET_ID = "CHEMBL203"  # EGFR
PCHEMBL_THRESHOLD = 7.0  # pIC50 >= 7.0 → active (IC50 <= 100 nM)
MW_MIN = 100
MW_MAX = 900


# ── Step 1: ChEMBL API에서 원시 데이터 추출 ────────────────────────────

def fetch_raw_data(target_id: str) -> pd.DataFrame:
    print(f"[Step 1] Fetching IC50 data for {target_id} from ChEMBL REST API...")

    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    params = {
        "target_chembl_id": target_id,
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_units": "nM",
        "limit": 1000,
        "offset": 0,
    }

    all_records = []
    while True:
        resp = requests.get(base_url, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        activities = data.get("activities", [])
        all_records.extend(activities)
        print(f"  → fetched {len(all_records)} records...", end="\r")

        next_url = data.get("page_meta", {}).get("next")
        if not next_url:
            break
        params["offset"] += params["limit"]
        time.sleep(0.3)  # rate limiting

    print(f"  → fetched {len(all_records)} records total")
    df = pd.DataFrame(all_records)

    cols = [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "standard_units",
        "standard_type",
        "standard_relation",
        "pchembl_value",
        "assay_chembl_id",
        "assay_type",
        "data_validity_comment",
    ]
    df = df[[c for c in cols if c in df.columns]]

    raw_path = RAW_DIR / "raw_egfr_ic50.csv"
    df.to_csv(raw_path, index=False)
    print(f"  → Raw data: {len(df)} rows → {raw_path}")
    return df


# ── Step 2: 기본 필터링 ──────────────────────────────────────────────

def basic_filter(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)

    # data_validity_comment가 있으면 문제 있는 데이터
    df = df[df["data_validity_comment"].isna()]
    n_after_validity = len(df)

    # pchembl_value가 있는 것만 (ChEMBL이 품질 검증한 값)
    df = df[df["pchembl_value"].notna()]
    df["pchembl_value"] = df["pchembl_value"].astype(float)
    n_after_pchembl = len(df)

    print(f"[Step 2] Basic filtering: {n_before} → {n_after_pchembl}")
    print(f"  → validity comment 제거: -{n_before - n_after_validity}")
    print(f"  → pchembl_value 없음 제거: -{n_after_validity - n_after_pchembl}")
    return df


# ── Step 3: SMILES 정리 ──────────────────────────────────────────────

def clean_smiles_col(df: pd.DataFrame) -> pd.DataFrame:
    remover = SaltRemover()

    def _clean(smi):
        if not smi or not isinstance(smi, str):
            return None
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol = remover.StripMol(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(mol)

    n_before = len(df)
    df = df.copy()
    df["clean_smiles"] = df["canonical_smiles"].apply(_clean)
    df = df[df["clean_smiles"].notna()]
    print(f"[Step 3] SMILES cleaning: {n_before} → {len(df)} (-{n_before - len(df)})")
    return df


# ── Step 4: 중복 처리 ────────────────────────────────────────────────

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)

    measurement_counts = (
        df.groupby("clean_smiles").size().reset_index(name="n_measurements")
    )
    df_dedup = (
        df.groupby("clean_smiles")
        .agg({"pchembl_value": "median", "molecule_chembl_id": "first"})
        .reset_index()
    )
    df_dedup = df_dedup.merge(measurement_counts, on="clean_smiles")

    print(f"[Step 4] Deduplication: {n_before} → {len(df_dedup)} unique molecules")
    print(
        f"  → Median measurements per molecule: {df_dedup['n_measurements'].median():.1f}"
    )
    return df_dedup


# ── Step 5: Label 생성 ───────────────────────────────────────────────

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = (df["pchembl_value"] >= PCHEMBL_THRESHOLD).astype(int)
    n_active = df["label"].sum()
    n_inactive = len(df) - n_active
    ratio = n_active / len(df) * 100
    print(f"[Step 5] Labels (threshold pIC50 >= {PCHEMBL_THRESHOLD}):")
    print(f"  → Active: {n_active} ({ratio:.1f}%)")
    print(f"  → Inactive: {n_inactive} ({100 - ratio:.1f}%)")
    print(f"  → Imbalance ratio: 1:{n_inactive / max(n_active, 1):.1f}")
    return df


# ── Step 6: 분자 크기 필터 ────────────────────────────────────────────

def filter_by_weight(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df.copy()
    df["mol_weight"] = df["clean_smiles"].apply(
        lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s))
        if Chem.MolFromSmiles(s)
        else None
    )
    df = df[df["mol_weight"].notna()]
    df = df[(df["mol_weight"] >= MW_MIN) & (df["mol_weight"] <= MW_MAX)]
    print(f"[Step 6] MW filter ({MW_MIN}–{MW_MAX}): {n_before} → {len(df)} (-{n_before - len(df)})")
    return df


# ── Step 7: Scaffold 계산 + 통계 리포트 ───────────────────────────────

def add_scaffolds(df: pd.DataFrame) -> pd.DataFrame:
    def _scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf)

    df = df.copy()
    df["scaffold"] = df["clean_smiles"].apply(_scaffold)
    return df


def print_report(df: pd.DataFrame):
    print()
    print("=" * 55)
    print("  DATASET STATISTICS REPORT")
    print("=" * 55)
    print(f"  Target:            EGFR ({TARGET_ID})")
    print(f"  Total molecules:   {len(df)}")
    print(f"  Active:            {df['label'].sum()} ({df['label'].mean() * 100:.1f}%)")
    print(f"  Inactive:          {(1 - df['label']).sum().astype(int)}")
    print(f"  Unique scaffolds:  {df['scaffold'].nunique()}")
    print(f"  MW:                {df['mol_weight'].mean():.1f} ± {df['mol_weight'].std():.1f}")
    print(f"  pIC50 range:       {df['pchembl_value'].min():.2f} – {df['pchembl_value'].max():.2f}")
    print(f"  pIC50 median:      {df['pchembl_value'].median():.2f}")
    print("=" * 55)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print()
    df = fetch_raw_data(TARGET_ID)
    df = basic_filter(df)
    df = clean_smiles_col(df)
    df = deduplicate(df)
    df = add_labels(df)
    df = filter_by_weight(df)
    df = add_scaffolds(df)

    # 최종 컬럼 정리 + 저장
    out_cols = [
        "molecule_chembl_id",
        "clean_smiles",
        "pchembl_value",
        "label",
        "mol_weight",
        "scaffold",
        "n_measurements",
    ]
    df = df[out_cols].reset_index(drop=True)

    out_path = PROCESSED_DIR / "egfr_cleaned.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[Done] Saved → {out_path}")

    print_report(df)


if __name__ == "__main__":
    main()
