## Error Analysis: ECFP+RF (Scaffold Split)

- TP: 285, TN: 491, FP: 140, FN: 118
- Accuracy: 0.750

### Molecular Properties: Error vs Correct

Property               Error mean   Correct mean    Diff%
----------------------------------------------------------
MolWt                      513.96         458.86   +12.0%
MolLogP                      4.76           4.54    +4.8%
TPSA                       100.46          92.17    +9.0%
NumAromaticRings             3.87           3.78    +2.4%

---

## Error Analysis: GCN-256 (Scaffold Split)

- TP: 297, TN: 465, FP: 166, FN: 106
- Accuracy: 0.737

### Molecular Properties: Error vs Correct

Property               Error mean   Correct mean    Diff%
----------------------------------------------------------
MolWt                      471.85         472.88    -0.2%
MolLogP                      4.70           4.56    +3.1%
TPSA                        90.46          95.58    -5.4%
NumAromaticRings             3.83           3.79    +1.2%

---

## Distance to Training Set vs Performance

Tanimoto Range         n   ECFP+RF AUC   GCN-256 AUC
----------------------------------------------------
[0.0, 0.3)              84        0.5130        0.4097
[0.3, 0.5)             173        0.4767        0.5321
[0.5, 0.7)             289        0.8155        0.8085
[0.7, 1.0)             488        0.8613        0.8606

---

## Uncertainty Analysis (GCN-256 MC Dropout)

- Scaffold frequency vs uncertainty correlation: nan
  (Negative = rare scaffolds have higher uncertainty = expected)