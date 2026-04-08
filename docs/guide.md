# ChEMBL 기반 타깃 활성 예측 + Virtual Screening GNN 프로젝트 상세 가이드

---

## 0. 프로젝트 요약

이 프로젝트의 핵심 질문은 하나다:

> **"GNN이 분자의 그래프 구조를 직접 학습하면, fingerprint 기반 모델보다 새로운 화학 골격(scaffold)에 대해 더 잘 일반화할 수 있는가?"**

이 질문에 답하기 위해 다음을 수행한다:

- 특정 타깃 하나에 대한 inhibitor activity 데이터를 수집·정제한다
- ECFP 기반 전통 모델과 GNN 모델을 동일 조건에서 비교한다
- scaffold split을 통해 "진짜 일반화"를 측정한다
- top-k enrichment로 virtual screening 실용성을 평가한다
- 실패 사례와 uncertainty를 분석해서 모델의 한계를 정직하게 보고한다

---

## 1. 타깃 선정 가이드

### 1.1 좋은 타깃의 조건

타깃을 고를 때 아래 기준을 모두 충족하는 것이 이상적이다:

| 기준 | 최소 권장 | 이유 |
|------|----------|------|
| 데이터 수 | IC50 기준 2,000개 이상 | scaffold split 시 train/test 모두 충분해야 한다 |
| 측정 타입 통일성 | IC50가 전체의 70% 이상 | assay heterogeneity를 줄인다 |
| 문헌/배경 | 리뷰 논문이 존재 | 결과 해석과 보고서 작성이 쉬워진다 |
| 화학적 다양성 | scaffold 종류 500개 이상 | scaffold split이 의미를 가진다 |

### 1.2 추천 타깃 목록

데이터 양과 품질 측면에서 다음 타깃들이 첫 프로젝트에 적합하다:

**Tier 1 (데이터 매우 풍부, 5000+)**

- **EGFR (ChEMBL Target ID: CHEMBL203)** — 비소세포폐암 등의 핵심 타깃. IC50 데이터가 매우 많고, erlotinib/gefitinib 등 승인약이 있어 결과 해석이 쉽다.
- **DRD2 (CHEMBL217)** — 도파민 수용체. 정신질환 약물의 대표 타깃. 데이터가 풍부하고 화학적 다양성이 높다.
- **JAK2 (CHEMBL2971)** — 자가면역질환/혈액암 타깃. kinase 계열이라 EGFR과 구조적 유사성이 있어 transfer 실험도 가능하다.

**Tier 2 (데이터 충분, 2000–5000)**

- **BACE1 (CHEMBL4822)** — 알츠하이머 관련. MoleculeNet에 벤치마크가 이미 있어 비교가 용이하다.
- **hERG (CHEMBL240)** — 심장 독성 관련 채널. safety 예측이라는 점에서 실용적 의미가 크다.

**첫 프로젝트에서 피해야 할 타깃:**

- 데이터가 1000개 미만인 타깃 (scaffold split 시 test set이 너무 작아진다)
- GPCR 중 데이터가 적은 것들 (orphan receptor 등)
- multi-subunit complex 타깃 (어떤 subunit 데이터인지 혼란)

### 1.3 ChEMBL에서 타깃별 데이터 규모 확인법

프로젝트 시작 전에 반드시 해야 할 작업이다. ChEMBL 웹사이트에서 수동으로 확인할 수 있지만, Python으로 자동화하면 여러 타깃을 빠르게 비교할 수 있다:

```python
from chembl_webresource_client.new_client import new_client

target = new_client.target
activity = new_client.activity

# EGFR 예시
target_id = 'CHEMBL203'

# 해당 타깃의 IC50 데이터 수 확인
acts = activity.filter(
    target_chembl_id=target_id,
    standard_type='IC50',
    standard_relation='=',
    standard_units='nM'
)

# 리스트로 변환하여 카운트
act_list = list(acts)
print(f"Target: {target_id}")
print(f"IC50 data points (exact, nM): {len(act_list)}")
```

여기서 주의할 점: `standard_relation='='`은 정확한 수치만 가져온다. `'>'`나 `'<'`가 붙은 censored data는 별도로 처리해야 한다.

---

## 2. 데이터 수집과 정제 — 상세 파이프라인

### 2.1 ChEMBL에서 원시 데이터 추출

```python
import pandas as pd
from chembl_webresource_client.new_client import new_client

activity = new_client.activity

target_id = 'CHEMBL203'  # EGFR

# 필터: IC50만, 정확한 값만, nM 단위
results = activity.filter(
    target_chembl_id=target_id,
    standard_type='IC50',
    standard_relation='=',
    standard_units='nM'
)

df = pd.DataFrame.from_records(results)

# 필요한 컬럼만 추출
cols = [
    'molecule_chembl_id',
    'canonical_smiles',
    'standard_value',       # IC50 값 (nM)
    'standard_units',
    'standard_type',
    'standard_relation',
    'pchembl_value',         # -log10(IC50) — ChEMBL이 이미 계산해 놓은 값
    'assay_chembl_id',
    'assay_type',            # 'B' = binding, 'F' = functional 등
    'data_validity_comment'  # 데이터 품질 코멘트
]

df = df[[c for c in cols if c in df.columns]]
df.to_csv('raw_egfr_ic50.csv', index=False)
print(f"Raw data: {len(df)} rows")
```

### 2.2 정제 단계별 상세

아래 순서대로 진행한다. 각 단계에서 제거된 데이터 수를 기록해야 한다.

#### Step 1: 기본 필터링

```python
# data_validity_comment가 있으면 대부분 문제가 있는 데이터
df = df[df['data_validity_comment'].isna()]

# pchembl_value가 있는 것만 사용 (ChEMBL이 품질 검증한 값)
df = df[df['pchembl_value'].notna()]
df['pchembl_value'] = df['pchembl_value'].astype(float)
```

`pchembl_value`가 있다는 것은 ChEMBL 큐레이터가 해당 값을 신뢰할 만하다고 판단했다는 뜻이다. 이 필터 하나만으로 데이터 품질이 크게 올라간다.

#### Step 2: SMILES 정리

```python
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

remover = SaltRemover()

def clean_smiles(smi):
    """SMILES 정리: 파싱 → salt 제거 → canonical화"""
    if not smi or not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # salt 제거 (counterion 등)
    mol = remover.StripMol(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    # canonical SMILES로 변환
    return Chem.MolToSmiles(mol)

df['clean_smiles'] = df['canonical_smiles'].apply(clean_smiles)
n_before = len(df)
df = df[df['clean_smiles'].notna()]
print(f"SMILES cleaning: {n_before} → {len(df)} ({n_before - len(df)} removed)")
```

salt 제거가 왜 중요한가: 예를 들어 `[Na+].[O-]C(=O)c1ccccc1`라는 SMILES가 있으면, 실제 활성 분자는 벤조산이고 나트륨 이온은 counterion에 불과하다. 이걸 그대로 두면 그래프에 분리된 컴포넌트가 생겨서 모델이 혼란을 겪는다.

#### Step 3: 중복 처리

```python
# 같은 분자에 대한 여러 측정값 → 중앙값 사용
df_dedup = df.groupby('clean_smiles').agg({
    'pchembl_value': 'median',
    'molecule_chembl_id': 'first'
}).reset_index()

# 측정 횟수 기록 (나중에 분석용)
measurement_counts = df.groupby('clean_smiles').size().reset_index(name='n_measurements')
df_dedup = df_dedup.merge(measurement_counts, on='clean_smiles')

print(f"After dedup: {len(df_dedup)} unique molecules")
print(f"Median measurements per molecule: {df_dedup['n_measurements'].median():.1f}")
```

중앙값을 쓰는 이유: 같은 분자라도 다른 실험실, 다른 assay에서 측정하면 값이 다르다. 평균은 극단적 outlier에 취약하지만, 중앙값은 더 robust하다.

#### Step 4: label 생성

```python
THRESHOLD = 6.0  # pIC50 >= 6.0 → active (IC50 <= 1 μM)

df_dedup['label'] = (df_dedup['pchembl_value'] >= THRESHOLD).astype(int)

n_active = df_dedup['label'].sum()
n_inactive = len(df_dedup) - n_active
ratio = n_active / len(df_dedup) * 100

print(f"Active: {n_active} ({ratio:.1f}%)")
print(f"Inactive: {n_inactive} ({100-ratio:.1f}%)")
print(f"Imbalance ratio: 1:{n_inactive/n_active:.1f}")
```

threshold 선택에 대해: pIC50 = 6은 IC50 = 1 μM에 해당하며, 이는 medicinal chemistry에서 가장 흔히 사용되는 active/inactive 경계다. 하지만 타깃에 따라 5.0(10 μM)이나 7.0(100 nM)이 더 적절할 수도 있다. 해당 타깃의 known drug들의 potency 범위를 참고하면 좋다.

#### Step 5: 분자 크기 필터

```python
from rdkit.Chem import Descriptors

def get_mol_weight(smi):
    mol = Chem.MolFromSmiles(smi)
    return Descriptors.MolWt(mol) if mol else None

df_dedup['mol_weight'] = df_dedup['clean_smiles'].apply(get_mol_weight)

# 너무 작거나 너무 큰 분자 제거
# fragment나 polymer 같은 것들이 섞여 있을 수 있음
df_dedup = df_dedup[
    (df_dedup['mol_weight'] >= 100) &
    (df_dedup['mol_weight'] <= 900)
]
```

### 2.3 데이터 통계 리포트

정제가 끝나면 반드시 아래 통계를 기록한다:

```python
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

df_dedup['scaffold'] = df_dedup['clean_smiles'].apply(get_scaffold)

print("=" * 50)
print("DATASET STATISTICS REPORT")
print("=" * 50)
print(f"Total molecules: {len(df_dedup)}")
print(f"Active: {df_dedup['label'].sum()} ({df_dedup['label'].mean()*100:.1f}%)")
print(f"Inactive: {(1-df_dedup['label']).sum().astype(int)}")
print(f"Unique scaffolds: {df_dedup['scaffold'].nunique()}")
print(f"Molecular weight: {df_dedup['mol_weight'].mean():.1f} ± {df_dedup['mol_weight'].std():.1f}")
print(f"pIC50 range: {df_dedup['pchembl_value'].min():.2f} – {df_dedup['pchembl_value'].max():.2f}")
print(f"pIC50 median: {df_dedup['pchembl_value'].median():.2f}")
```

---

## 3. Split 설계 — 상세

### 3.1 Scaffold Split 구현

scaffold split의 핵심 아이디어: 같은 Murcko scaffold를 가진 분자들은 반드시 같은 세트(train 또는 test)에 들어가야 한다. 이렇게 하면 test set에는 training 때 한 번도 본 적 없는 화학 골격만 남게 된다.

```python
import numpy as np
from collections import defaultdict

def scaffold_split(df, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42):
    """
    Scaffold 기반 데이터 split.
    같은 scaffold를 가진 분자들은 반드시 같은 세트에 들어간다.
    """
    np.random.seed(seed)

    # scaffold별 분자 인덱스 그룹핑
    scaffold_to_indices = defaultdict(list)
    for idx, row in df.iterrows():
        scaffold_to_indices[row['scaffold']].append(idx)

    # scaffold를 크기 순으로 정렬 (큰 그룹부터 배치)
    scaffold_sets = list(scaffold_to_indices.values())
    scaffold_sets.sort(key=len, reverse=True)

    n_total = len(df)
    n_train = int(n_total * frac_train)
    n_valid = int(n_total * frac_valid)

    train_idx, valid_idx, test_idx = [], [], []

    for group in scaffold_sets:
        if len(train_idx) + len(group) <= n_train:
            train_idx.extend(group)
        elif len(valid_idx) + len(group) <= n_valid:
            valid_idx.extend(group)
        else:
            test_idx.extend(group)

    return train_idx, valid_idx, test_idx
```

### 3.2 Random Split과의 비교 설정

```python
from sklearn.model_selection import train_test_split

def random_split(df, frac_train=0.8, frac_valid=0.1, seed=42):
    """Stratified random split"""
    train_idx, temp_idx = train_test_split(
        df.index, train_size=frac_train,
        stratify=df['label'], random_state=seed
    )
    valid_idx, test_idx = train_test_split(
        temp_idx, train_size=frac_valid/(1-frac_train),
        stratify=df.loc[temp_idx, 'label'], random_state=seed
    )
    return list(train_idx), list(valid_idx), list(test_idx)
```

### 3.3 Split 품질 검증

split을 만든 후 반드시 아래를 확인한다:

```python
def verify_split(df, train_idx, valid_idx, test_idx, split_name):
    """Split 품질 검증"""
    print(f"\n{'='*40}")
    print(f"Split: {split_name}")
    print(f"{'='*40}")

    for name, idx in [('Train', train_idx), ('Valid', valid_idx), ('Test', test_idx)]:
        subset = df.loc[idx]
        print(f"\n{name}:")
        print(f"  Size: {len(idx)} ({len(idx)/len(df)*100:.1f}%)")
        print(f"  Active ratio: {subset['label'].mean()*100:.1f}%")
        print(f"  Unique scaffolds: {subset['scaffold'].nunique()}")

    # scaffold split의 경우 train/test scaffold 겹침 확인
    train_scaffolds = set(df.loc[train_idx, 'scaffold'])
    test_scaffolds = set(df.loc[test_idx, 'scaffold'])
    overlap = train_scaffolds & test_scaffolds
    print(f"\nScaffold overlap (train ∩ test): {len(overlap)}")

    if split_name == 'scaffold' and len(overlap) > 0:
        print("⚠️ WARNING: Scaffold split has overlapping scaffolds!")
```

scaffold split에서 train/test의 scaffold 겹침이 0이어야 한다. 0이 아니면 구현에 버그가 있다는 뜻이다.

---

## 4. Feature Engineering — 분자를 모델 입력으로 변환

### 4.1 ECFP (Extended Connectivity Fingerprint) — Baseline용

```python
from rdkit.Chem import AllChem

def smiles_to_ecfp(smi, radius=2, nbits=2048):
    """
    SMILES → ECFP 비트 벡터.
    radius=2 → ECFP4 (직경 4)
    radius=3 → ECFP6 (직경 6)
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp)

# 전체 데이터에 적용
X_ecfp = np.array([smiles_to_ecfp(s) for s in df_dedup['clean_smiles']])
```

ECFP가 왜 강력한 baseline인가: ECFP는 각 원자를 중심으로 반경 r 이내의 원자 환경을 해싱하여 비트 벡터로 변환한다. 이 과정이 사실상 "로컬 분자 그래프의 요약"이다. GNN의 message passing과 개념적으로 유사하지만, 학습 없이 고정된 해시 함수를 쓴다는 차이가 있다. GNN이 ECFP보다 나으려면, 이 고정된 해싱보다 더 좋은 표현을 학습으로 찾아야 한다.

### 4.2 분자 그래프 변환 — GNN용

```python
import torch
from torch_geometric.data import Data
from rdkit import Chem

# 원자 feature 정의
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),         # 원소 번호
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'num_hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

def one_hot(value, allowable_set):
    """값을 one-hot 인코딩. 미등록 값은 마지막 비트를 켠다."""
    encoding = [0] * (len(allowable_set) + 1)
    if value in allowable_set:
        encoding[allowable_set.index(value)] = 1
    else:
        encoding[-1] = 1  # unknown
    return encoding

def atom_features(atom):
    """원자 하나의 feature 벡터 생성"""
    features = []
    features += one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    features += one_hot(atom.GetDegree(), ATOM_FEATURES['degree'])
    features += one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    features += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs'])
    features += one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(1 if atom.IsInRing() else 0)
    return features

def bond_features(bond):
    """결합 하나의 feature 벡터 생성"""
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
    features += one_hot(stereo, [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ])
    return [float(f) for f in features]

def smiles_to_graph(smi, label):
    """SMILES → PyTorch Geometric Data 객체"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    # 노드 features
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    # 엣지 (양방향)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        # 양방향 추가
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bf)
        edge_attr.append(bf)

    if len(edge_index) == 0:
        # 단일 원자 분자 (예: noble gas) — 엣지 없음
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, len(bf)), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    y = torch.tensor([label], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
```

각 feature가 왜 필요한가:

- **atomic number**: 원소 종류 — 가장 기본적인 화학 정보
- **degree**: 결합 수 — 원자의 연결 패턴
- **formal charge**: 이온화 상태 — 전하를 띤 원자는 약물 상호작용에 큰 영향
- **hybridization**: sp, sp2, sp3 등 — 분자의 3D 형태에 영향
- **aromaticity**: 방향족 여부 — 약물 분자의 핵심 구조 요소
- **ring membership**: 고리 포함 여부 — 약물의 rigidity/flexibility에 영향
- **bond type**: 단일/이중/삼중/방향족 결합 — 분자의 전자 구조
- **conjugation**: 공액 여부 — 전자 비편재화, 반응성에 영향

---

## 5. Baseline 모델 — 상세 구현

### 5.1 ECFP + XGBoost

```python
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

def train_xgboost_baseline(X_train, y_train, X_valid, y_valid):
    """ECFP 기반 XGBoost 학습"""

    # 클래스 불균형 처리
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        eval_metric='aucpr',
        early_stopping_rounds=30,
        random_state=42,
        use_label_encoder=False
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=50
    )

    return model
```

`scale_pos_weight` 파라미터가 중요하다. active/inactive 비율이 1:5라면 이 값을 5로 설정한다. 이렇게 하면 모델이 소수 클래스(active)의 오분류에 더 큰 페널티를 부여한다.

### 5.2 ECFP + Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

def train_rf_baseline(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        class_weight='balanced',    # 자동 불균형 보정
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
```

### 5.3 MLP on ECFP

```python
import torch.nn as nn

class FingerprintMLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout)
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)
```

이 MLP가 중요한 이유: GNN이 "그래프 구조를 직접 학습하는 것"의 이점을 주장하려면, 같은 정보를 fingerprint로 압축한 뒤 MLP로 학습한 것과 비교해야 한다. MLP가 이미 충분히 좋다면, GNN의 추가 복잡성이 정당화되지 않을 수 있다.

---

## 6. GNN 모델 — 상세 구현

### 6.1 GCN (Graph Convolutional Network)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool

class GCNModel(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=256, num_layers=4, dropout=0.2):
        super().__init__()

        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout

        # 두 가지 pooling을 concatenate
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 노드 임베딩
        x = self.node_encoder(x)

        # Message passing layers
        for conv, bn in zip(self.convs, self.bns):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # residual connection

        # Graph-level readout: mean + sum
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x_graph = torch.cat([x_mean, x_sum], dim=-1)

        return self.head(x_graph).squeeze(-1)
```

설계 선택에 대한 설명:

- **Residual connection**: 깊은 GNN에서 over-smoothing(모든 노드의 표현이 비슷해지는 현상)을 완화한다. 4층 이상일 때 특히 중요하다.
- **Mean + Sum pooling 결합**: mean pooling은 분자 크기에 불변하고, sum pooling은 크기 정보를 보존한다. 둘을 합치면 더 풍부한 graph-level 표현을 얻는다.
- **BatchNorm**: 학습 안정성을 크게 개선한다. 분자마다 크기가 다르기 때문에 feature 스케일이 달라질 수 있는데, BN이 이를 보정한다.

### 6.2 GIN (Graph Isomorphism Network)

```python
from torch_geometric.nn import GINConv

class GINModel(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=256, num_layers=5, dropout=0.2):
        super().__init__()

        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            # GIN의 MLP: 2층 + BN
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout

        # JK (Jumping Knowledge): 모든 레이어의 표현을 합산
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * (num_layers + 1), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.num_layers = num_layers

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

        # Jumping Knowledge: 모든 레이어의 graph-level 표현을 concatenate
        x_graph = torch.cat(layer_outputs, dim=-1)

        return self.head(x_graph).squeeze(-1)
```

GIN이 GCN보다 이론적으로 강한 이유: GIN은 Weisfeiler-Leman (WL) graph isomorphism test와 동일한 표현력을 가진다고 증명되었다. 즉, WL test로 구분할 수 있는 두 그래프는 GIN도 구분할 수 있다. GCN은 이보다 약하다.

Jumping Knowledge (JK)의 역할: 마지막 레이어의 표현만 쓰면, 멀리 있는 원자의 정보만 과도하게 반영된다. JK는 모든 레이어의 표현을 합쳐서, 로컬(얕은 레이어)과 글로벌(깊은 레이어) 정보를 모두 활용한다.

### 6.3 D-MPNN (Directed Message Passing Neural Network)

```python
from torch_geometric.nn import NNConv

class DMPNNModel(nn.Module):
    """
    엣지 feature를 활용하는 Message Passing 모델.
    Chemprop 스타일의 directed message passing을 간소화한 버전.
    """
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=256,
                 num_layers=4, dropout=0.2):
        super().__init__()

        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            # edge feature → message weight matrix
            edge_nn = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden_dim * hidden_dim),
            )
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr='mean'))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
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
```

D-MPNN이 중요한 이유: 일반 GCN/GIN은 edge feature를 활용하지 않거나 제한적으로만 사용한다. 하지만 화학에서 결합 종류(단일, 이중, 방향족 등)는 매우 중요한 정보다. D-MPNN은 이 정보를 message passing 과정에 직접 통합한다.

---

## 7. 학습 루프 — 상세

### 7.1 Training Loop

```python
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

def train_gnn(model, train_loader, valid_loader, device,
              lr=1e-3, weight_decay=1e-5, epochs=200, patience=20):
    """
    GNN 학습 루프 (early stopping 포함)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
    )

    # 클래스 불균형 처리
    # train_loader에서 positive 비율 계산
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch.y.tolist())
    pos_weight = torch.tensor([(len(all_labels) - sum(all_labels)) / sum(all_labels)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    best_valid_auc = 0
    best_epoch = 0
    train_history = []
    valid_history = []

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            # gradient clipping — 학습 안정성
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_loss / len(train_loader.dataset)

        # --- Validate ---
        valid_auc, valid_ap = evaluate_gnn(model, valid_loader, device)

        scheduler.step(valid_auc)
        train_history.append(avg_train_loss)
        valid_history.append(valid_auc)

        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pt')

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch}. Best: {best_epoch}")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {avg_train_loss:.4f} | "
                  f"Valid AUC: {valid_auc:.4f} | Valid AP: {valid_ap:.4f}")

    # 최고 모델 로드
    model.load_state_dict(torch.load('best_model.pt'))
    return model, train_history, valid_history


def evaluate_gnn(model, loader, device):
    """모델 평가: AUC, AP 계산"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            probs = torch.sigmoid(out)
            all_preds.extend(probs.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    auc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)
    return auc, ap
```

### 7.2 Hyperparameter 탐색 가이드

처음 프로젝트에서 전부를 탐색할 필요는 없다. 아래 우선순위로 진행한다:

**반드시 시도해야 할 것 (영향 큼):**

| 파라미터 | 탐색 범위 | 기본값 |
|---------|----------|-------|
| learning rate | {1e-4, 5e-4, 1e-3} | 1e-3 |
| hidden dim | {128, 256} | 256 |
| num layers | {3, 4, 5} | 4 |
| dropout | {0.1, 0.2, 0.3} | 0.2 |

**시간 있으면 시도할 것 (영향 중간):**

| 파라미터 | 탐색 범위 | 기본값 |
|---------|----------|-------|
| batch size | {32, 64, 128} | 64 |
| weight decay | {1e-6, 1e-5, 1e-4} | 1e-5 |
| pooling | {mean, sum, mean+sum} | mean+sum |

---

## 8. Evaluation — 상세 메트릭 구현

### 8.1 기본 분류 메트릭

```python
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)

def full_evaluation(y_true, y_pred_proba, threshold=0.5):
    """포괄적 평가 메트릭 계산"""
    results = {}

    # 기본 메트릭
    results['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    results['pr_auc'] = average_precision_score(y_true, y_pred_proba)

    # threshold 기반 메트릭
    y_pred = (np.array(y_pred_proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    results['confusion_matrix'] = cm

    return results
```

### 8.2 Virtual Screening 메트릭 — 핵심

```python
def enrichment_factor(y_true, y_pred_proba, percentage):
    """
    Enrichment Factor 계산.

    EF = (상위 x%에서의 active 비율) / (전체에서의 active 비율)

    예: EF1% = 20이면, 상위 1%에서 active가 전체 대비 20배 더 많다는 뜻.
    랜덤 선택 대비 20배 효율적이라는 의미.
    """
    n = len(y_true)
    n_top = max(1, int(n * percentage / 100))

    # 예측 점수 기준 정렬
    sorted_indices = np.argsort(y_pred_proba)[::-1]  # 높은 점수부터

    # 상위 x%에서의 active 수
    top_actives = np.array(y_true)[sorted_indices[:n_top]].sum()

    # EF 계산
    total_actives = np.array(y_true).sum()
    if total_actives == 0:
        return 0.0

    ef = (top_actives / n_top) / (total_actives / n)
    return ef


def top_k_precision(y_true, y_pred_proba, k):
    """상위 k개 중 실제 active의 비율"""
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    top_k_labels = np.array(y_true)[sorted_indices[:k]]
    return top_k_labels.sum() / k


def screening_evaluation(y_true, y_pred_proba):
    """Virtual screening 관점의 종합 평가"""
    results = {}

    for pct in [1, 2, 5, 10]:
        results[f'EF{pct}%'] = enrichment_factor(y_true, y_pred_proba, pct)

    for k in [10, 20, 50, 100]:
        if k <= len(y_true):
            results[f'Precision@{k}'] = top_k_precision(y_true, y_pred_proba, k)

    return results
```

EF1%의 직관적 해석: "만약 라이브러리에서 상위 1%만 실험한다면, 랜덤으로 1% 골랐을 때보다 몇 배나 더 많은 hit을 찾는가?"

EF1% = 20이면, 모델이 상위 1%를 아주 잘 뽑아서 랜덤 대비 20배 효율적이라는 뜻이다. 실전에서 이 수치는 매우 중요하다. 화합물 합성과 실험 비용이 비싸기 때문에, 가능한 적은 수의 후보만 선별해서 높은 hit rate를 얻는 것이 핵심이다.

---

## 9. Uncertainty Estimation — 상세 구현

### 9.1 Deep Ensemble

```python
def train_ensemble(model_class, model_kwargs, train_loader, valid_loader,
                   device, n_models=5, **train_kwargs):
    """N개의 독립 모델을 학습하여 ensemble 구성"""
    models = []
    for i in range(n_models):
        print(f"\n--- Training ensemble member {i+1}/{n_models} ---")

        # 각 모델은 다른 random seed로 초기화
        torch.manual_seed(42 + i)
        model = model_class(**model_kwargs).to(device)
        model, _, _ = train_gnn(model, train_loader, valid_loader, device, **train_kwargs)
        models.append(model)

    return models


def ensemble_predict(models, loader, device):
    """Ensemble 예측: 평균과 불확실성 계산"""
    all_member_preds = []

    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                probs = torch.sigmoid(out)
                preds.extend(probs.cpu().tolist())
        all_member_preds.append(preds)

    all_member_preds = np.array(all_member_preds)  # (n_models, n_samples)

    # 앙상블 평균 = 최종 예측
    mean_pred = all_member_preds.mean(axis=0)

    # 앙상블 표준편차 = epistemic uncertainty 추정
    std_pred = all_member_preds.std(axis=0)

    return mean_pred, std_pred
```

### 9.2 MC Dropout

```python
def mc_dropout_predict(model, loader, device, n_forward=30):
    """MC Dropout: training mode에서 여러 번 forward하여 불확실성 추정"""
    model.train()  # dropout 활성화 상태 유지

    all_preds = []
    for _ in range(n_forward):
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                probs = torch.sigmoid(out)
                preds.extend(probs.cpu().tolist())
        all_preds.append(preds)

    all_preds = np.array(all_preds)  # (n_forward, n_samples)

    mean_pred = all_preds.mean(axis=0)
    std_pred = all_preds.std(axis=0)

    return mean_pred, std_pred
```

### 9.3 Uncertainty 분석

```python
def uncertainty_analysis(y_true, mean_pred, std_pred, scaffolds_test):
    """불확실성 기반 분석"""

    # 1. 확신도 기반 필터링: 상위 x% 확신 분자만 평가
    print("\n=== Confidence-based filtering ===")
    for pct in [100, 75, 50, 25]:
        threshold = np.percentile(std_pred, pct)  # std가 낮을수록 확신
        mask = std_pred <= threshold
        if mask.sum() > 10:
            subset_auc = roc_auc_score(
                np.array(y_true)[mask],
                mean_pred[mask]
            )
            print(f"Top {pct}% confident ({mask.sum()} molecules): AUC = {subset_auc:.4f}")

    # 2. Scaffold novelty vs uncertainty 상관관계
    # 새로운 scaffold일수록 불확실성이 높은가?
    unique_scaffolds = list(set(scaffolds_test))
    scaffold_counts_test = pd.Series(scaffolds_test).value_counts()

    # 빈도가 낮은 scaffold = 더 "새로운" scaffold
    scaffold_freq = [scaffold_counts_test.get(s, 0) for s in scaffolds_test]
    correlation = np.corrcoef(scaffold_freq, std_pred)[0, 1]
    print(f"\nCorrelation (scaffold frequency vs uncertainty): {correlation:.3f}")
    print("(음수면: 드문 scaffold일수록 불확실성이 높다 = 기대한 대로)")
```

이 분석이 왜 가치 있는가: 단순히 "AUC = 0.85"보다 "모델이 자신 있어하는 상위 50% 분자만 보면 AUC = 0.93이고, 나머지 50%에서는 AUC = 0.71"이라는 결과가 훨씬 실용적이다. 이는 medicinal chemist에게 "이 후보는 모델이 확신하니 우선 진행하고, 저 후보는 추가 실험이 필요하다"고 말할 수 있게 한다.

---

## 10. Error Analysis — 상세 프레임워크

### 10.1 체계적 오류 분류

```python
def error_analysis(df_test, y_true, y_pred_proba, threshold=0.5):
    """체계적 오류 분석"""

    y_pred = (np.array(y_pred_proba) >= threshold).astype(int)
    y_true = np.array(y_true)

    # 오류 유형 분류
    false_positives = np.where((y_pred == 1) & (y_true == 0))[0]
    false_negatives = np.where((y_pred == 0) & (y_true == 1))[0]
    true_positives = np.where((y_pred == 1) & (y_true == 1))[0]
    true_negatives = np.where((y_pred == 0) & (y_true == 0))[0]

    print(f"True Positives:  {len(true_positives)}")
    print(f"True Negatives:  {len(true_negatives)}")
    print(f"False Positives: {len(false_positives)} (모델이 active로 예측했지만 실제론 inactive)")
    print(f"False Negatives: {len(false_negatives)} (모델이 inactive로 예측했지만 실제론 active)")

    return {
        'FP_indices': false_positives,
        'FN_indices': false_negatives,
        'TP_indices': true_positives,
        'TN_indices': true_negatives
    }
```

### 10.2 분자 속성별 오류 패턴

```python
from rdkit.Chem import Descriptors, Fragments

def property_based_error_analysis(df_test, error_indices, correct_indices):
    """어떤 분자 속성에서 오류가 집중되는지 분석"""

    # 분자 속성 계산
    properties = ['MolWt', 'LogP', 'TPSA', 'NumRotatableBonds',
                  'NumHDonors', 'NumHAcceptors', 'NumAromaticRings']

    for prop_name in properties:
        func = getattr(Descriptors, prop_name, None)
        if func is None:
            continue

        error_vals = []
        correct_vals = []

        for idx in error_indices:
            mol = Chem.MolFromSmiles(df_test.iloc[idx]['clean_smiles'])
            if mol:
                error_vals.append(func(mol))

        for idx in correct_indices:
            mol = Chem.MolFromSmiles(df_test.iloc[idx]['clean_smiles'])
            if mol:
                correct_vals.append(func(mol))

        if error_vals and correct_vals:
            err_mean = np.mean(error_vals)
            cor_mean = np.mean(correct_vals)
            diff_pct = (err_mean - cor_mean) / cor_mean * 100 if cor_mean != 0 else 0
            print(f"{prop_name:25s} | Error mean: {err_mean:8.2f} | "
                  f"Correct mean: {cor_mean:8.2f} | Diff: {diff_pct:+.1f}%")
```

### 10.3 Scaffold 거리 vs 성능

```python
from rdkit.Chem import DataStructs, AllChem

def scaffold_distance_analysis(df_train, df_test, y_true_test, y_pred_test):
    """
    Test 분자가 training set과 얼마나 다른지 (Tanimoto 거리)와
    예측 정확도의 관계를 분석
    """

    # Training set의 ECFP 계산
    train_fps = []
    for smi in df_train['clean_smiles']:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            train_fps.append(fp)

    # 각 test 분자에 대해 가장 가까운 training 분자와의 Tanimoto 유사도
    max_similarities = []
    for smi in df_test['clean_smiles']:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            max_similarities.append(max(sims))
        else:
            max_similarities.append(0)

    # 유사도 구간별 성능
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    y_true_arr = np.array(y_true_test)
    y_pred_arr = np.array(y_pred_test)
    max_sim_arr = np.array(max_similarities)

    print("\n=== Performance by distance to training set ===")
    for low, high in bins:
        mask = (max_sim_arr >= low) & (max_sim_arr < high)
        n = mask.sum()
        if n > 20 and len(set(y_true_arr[mask])) > 1:
            auc = roc_auc_score(y_true_arr[mask], y_pred_arr[mask])
            print(f"Tanimoto [{low:.1f}, {high:.1f}): n={n:4d}, AUC={auc:.4f}")
        else:
            print(f"Tanimoto [{low:.1f}, {high:.1f}): n={n:4d}, (insufficient data)")
```

이 분석의 핵심 인사이트: training set과 유사한 분자(Tanimoto > 0.7)에서는 높은 AUC가 나오고, 거리가 멀어질수록(Tanimoto < 0.3) AUC가 급격히 떨어지는 것이 일반적이다. 이 "성능 감쇠 곡선"이 GNN과 ECFP 모델에서 어떻게 다른지 비교하면, GNN의 일반화 능력에 대한 실질적인 증거를 얻을 수 있다.

---

## 11. 실행 로드맵 — 주 단위 일정

### Phase 0: 환경 구축 (1주)

```bash
# 가상환경 생성
conda create -n drug_discovery python=3.10
conda activate drug_discovery

# 핵심 패키지
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch-geometric
pip install rdkit
pip install chembl_webresource_client
pip install xgboost scikit-learn pandas numpy matplotlib seaborn
pip install tdc  # Therapeutics Data Commons
```

**이 주에 확인할 것:**
- GPU 동작 확인: `torch.cuda.is_available()`
- RDKit 동작 확인: `from rdkit import Chem; Chem.MolFromSmiles('CCO')`
- ChEMBL API 접속 확인
- 간단한 PyG 그래프 생성 테스트

### Phase 1: 데이터 파이프라인 (2주)

**1주차:** 데이터 수집 + 정제
- ChEMBL에서 타깃 데이터 추출
- 정제 파이프라인 구현
- 통계 리포트 생성

**2주차:** Split + Feature 생성
- Random / Scaffold split 구현
- ECFP 생성
- 그래프 변환 구현
- Split 품질 검증

**산출물:**
- `data/cleaned_dataset.csv`
- `data/splits/random_train.npy`, `random_valid.npy`, `random_test.npy`
- `data/splits/scaffold_train.npy`, `scaffold_valid.npy`, `scaffold_test.npy`
- `reports/data_statistics.md`

### Phase 2: Baseline (1주)

- ECFP + XGBoost
- ECFP + Random Forest
- ECFP + MLP
- 각각 random split + scaffold split에서 평가

**산출물:**
- `results/baseline_performance.csv`
- `figures/baseline_roc_curves.png`
- `figures/baseline_pr_curves.png`

### Phase 3: GNN (2주)

**1주차:** GCN + GIN 구현 및 학습
**2주차:** 하이퍼파라미터 탐색 + D-MPNN 시도

**산출물:**
- `results/gnn_performance.csv`
- `results/all_models_comparison.csv`
- `figures/learning_curves.png`
- `figures/model_comparison_bar.png`

### Phase 4: Screening 분석 (1주)

- Top-k enrichment 계산
- Enrichment factor 그래프
- Top-ranked 후보 분자 시각화
- Scaffold diversity of top hits

**산출물:**
- `results/screening_metrics.csv`
- `figures/enrichment_curves.png`
- `figures/top_hits_structures.png`

### Phase 5: Error + Uncertainty 분석 (1주)

- Error analysis 전체 수행
- Ensemble uncertainty 학습 (또는 MC dropout)
- Confidence-filtered performance
- Scaffold distance vs performance

**산출물:**
- `results/error_analysis.md`
- `results/uncertainty_analysis.md`
- `figures/confidence_vs_performance.png`
- `figures/distance_vs_performance.png`

### Phase 6: 보고서 작성 (1주)

**산출물:**
- 최종 보고서 (10–15 페이지)
- 코드 정리 + README
- 재현 가능한 실행 스크립트

**총 기간: 약 8–9주**

---

## 12. 보고서 작성 가이드

### 12.1 권장 구조

**1. Introduction (1–2p)**
- 약물탐색에서 virtual screening의 역할
- GNN이 분자 그래프에 적용된 배경
- 이 프로젝트의 구체적 질문

**2. Dataset (1–2p)**
- 타깃 설명 및 선정 이유
- 데이터 수집 과정
- 정제 각 단계와 제거된 데이터 수
- 최종 통계 (표 1개, 그림 1–2개)

**3. Methods (2–3p)**
- Split 전략 (random vs scaffold)과 그 이유
- Baseline: ECFP + 전통 ML
- GNN: 구조, feature, 학습 세팅
- Evaluation metrics 설명

**4. Results (2–3p)**
- Baseline vs GNN 성능 비교 표
- Random split vs Scaffold split 비교
- Enrichment factor 그래프
- 학습 곡선

**5. Analysis (2–3p)**
- Error analysis: 어떤 분자에서 실패하는가
- Uncertainty analysis: 확신도와 정확도의 관계
- Scaffold distance vs 성능 관계

**6. Discussion (1p)**
- GNN이 실제로 이점을 보이는 조건
- 한계와 주의사항
- 실제 약물탐색에서의 의미

**7. Conclusion & Future Work (0.5p)**

### 12.2 핵심 테이블과 그림

프로젝트의 핵심 결과를 전달하는 데 반드시 필요한 시각화:

**Table 1: 전체 모델 비교**

| Model | Split | ROC-AUC | PR-AUC | EF1% | EF5% |
|-------|-------|---------|--------|------|------|
| ECFP+XGB | random | | | | |
| ECFP+XGB | scaffold | | | | |
| ECFP+MLP | random | | | | |
| ECFP+MLP | scaffold | | | | |
| GCN | random | | | | |
| GCN | scaffold | | | | |
| GIN | random | | | | |
| GIN | scaffold | | | | |

**Figure 1:** ROC/PR 곡선 (모든 모델, scaffold split)
**Figure 2:** Enrichment factor 곡선 (1%, 2%, 5%, 10%)
**Figure 3:** Scaffold distance vs AUC (ECFP vs GNN 비교)
**Figure 4:** Uncertainty-filtered performance curve

---

## 13. 흔한 실수와 대응

| 실수 | 증상 | 대응 |
|------|------|------|
| Data leakage | Random split AUC 0.95+ | Scaffold split 확인. 같은 분자가 train/test에 있는지 체크 |
| Label imbalance 무시 | 높은 accuracy, 낮은 PR-AUC | class weight 조정, focal loss, PR-AUC를 메인 지표로 |
| Assay 혼합 | 예측이 들쭉날쭉 | IC50만 사용, pchembl_value가 있는 것만 |
| Over-engineering | 학습 불안정, 코드 복잡 | 간단한 모델부터 시작. GCN 3층이면 충분 |
| 시각화 부족 | 결과 해석 어려움 | ROC, PR, enrichment, 학습곡선 반드시 생성 |
| Hyperparameter 과다 | 시간 소비, overfitting | 3–4개 핵심 파라미터만 탐색 |
| Reproducibility 무시 | 결과 재현 불가 | seed 고정, config 파일 사용, requirements.txt |

---

## 14. 확장 로드맵

### v2: Uncertainty + Active Learning

- Deep ensemble (5개) 추가
- Active learning 시뮬레이션: "불확실한 분자를 추가 학습하면 성능이 오르는가"
- Diversity-aware selection: 구조적 다양성을 고려한 후보 선정

### v3: Multi-task Learning

- 하나의 모델이 potency + toxicity + solubility를 동시에 예측
- Task 간 상관관계가 성능을 높이는지 확인
- Auxiliary task의 효과 분석

### v4: 3D 정보 활용

- RDKit `EmbedMolecule`로 3D conformer 생성
- SchNet 또는 DimeNet으로 3D 기반 모델 학습
- 2D vs 3D 비교

### v5: Protein-Ligand 모델

- Ligand-only → Target-aware로 전환
- Protein sequence 또는 구조 정보 결합
- 새로운 타깃에 대한 zero/few-shot 예측

---

## 15. 프로젝트의 핵심 메시지

이 프로젝트가 궁극적으로 보여줘야 하는 것은 "GNN이 좋다"가 아니다.

보여줘야 하는 것:

> 분자의 그래프 구조를 직접 학습하는 것이, 새로운 화학 공간에서의 예측에 실질적인 이점을 주는가? 그리고 그 이점은 어디에서 나타나고, 어디에서 사라지는가?

이 질문에 데이터로 답하면, 그것이 좋은 프로젝트다.

모델이 만능이 아님을 인정하고, 구체적으로 어떤 조건에서 유용한지를 밝히는 것. 그것이 이 프로젝트의 가치이자, 실제 약물탐색에서 computational method를 신뢰하는 방법이기도 하다.
