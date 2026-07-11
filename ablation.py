"""
Ablation study: marginal contribution of each feature layer in Model C.

Layer 1 — Fuzzy only         : 12 features (FuzzyGrader output)
Layer 2 — Fuzzy + Engineered : 12 + 8 = 20 features
Layer 3 — Fuzzy + Symbolic   : 12 + 9 = 21 features
Layer 4 — All (Model C)      : 12 + 8 + 9 = 29 features

All variants use identical XGBoost params (get_xgb_params_c) and
identical 10-fold CV split (RANDOM_STATE=42) for fair comparison.
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from src.data.loader import load_dataset
from src.grading.fuzzy_grader import FuzzyGrader
from src.grading.feature_engineer import FeatureEngineer
from src.symbolic.pipeline import SymbolicPipeline
from src.models.base import get_xgb_params_c, RANDOM_STATE, N_SPLITS, DISEASES


def cv_eval(X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    all_true, all_pred = [], []
    for train_idx, val_idx in cv.split(X, y):
        model = XGBClassifier(**get_xgb_params_c())
        model.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)
        pred = model.predict(X.iloc[val_idx])
        accs.append(accuracy_score(y.iloc[val_idx], pred))
        f1s.append(f1_score(y.iloc[val_idx], pred, average="macro", zero_division=0))
        all_true.extend(y.iloc[val_idx].tolist())
        all_pred.extend(pred.tolist())
    per_class = f1_score(all_true, all_pred, average=None, zero_division=0)
    return {
        "label": label,
        "n_features": X.shape[1],
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "per_class_f1": {DISEASES[i]: round(float(per_class[i]), 4) for i in range(len(DISEASES))},
    }


X_clinical, _, _, y = load_dataset()

grader = FuzzyGrader()
X_fuzzy = grader.grade(X_clinical).reset_index(drop=True)

engineer = FeatureEngineer()
X_eng = engineer.engineer(X_fuzzy)

pipeline = SymbolicPipeline("rules")
X_sym = pipeline.transform(X_fuzzy).reset_index(drop=True)

configs = [
    ("Fuzzy only          (12)",  X_fuzzy),
    ("Fuzzy + Engineered  (20)",  pd.concat([X_fuzzy, X_eng], axis=1)),
    ("Fuzzy + Symbolic    (21)",  pd.concat([X_fuzzy, X_sym], axis=1)),
    ("Fuzzy + Eng + Sym   (29)",  pd.concat([X_fuzzy, X_eng, X_sym], axis=1)),
]

results = [cv_eval(X, y, label) for label, X in configs]

print("\n" + "=" * 72)
print(f"{'Configuration':<35} {'Features':>8} {'Accuracy':>14} {'Macro F1':>12}")
print("-" * 72)
for r in results:
    print(f"{r['label']:<35} {r['n_features']:>8} "
          f"{r['acc_mean']:.4f} ±{r['acc_std']:.4f}  {r['f1_mean']:.4f} ±{r['f1_std']:.4f}")
print("=" * 72)

print("\nPer-class F1:")
print(f"{'Disease':<30}", end="")
for r in results:
    lbl = r['label'].split('(')[0].strip()[:12]
    print(f" {lbl:>12}", end="")
print()
print("-" * (30 + 13 * len(results)))
for d in DISEASES:
    print(f"{d:<30}", end="")
    for r in results:
        print(f" {r['per_class_f1'][d]:>12.4f}", end="")
    print()

print("\nMarginal gains (accuracy):")
for i in range(1, len(results)):
    delta = results[i]['acc_mean'] - results[i-1]['acc_mean']
    print(f"  {results[i-1]['label'].strip()} → {results[i]['label'].strip()}: {delta:+.4f}")
