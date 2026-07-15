"""
CatBoost trial on Model C's 29-feature pipeline.
Compares directly against XGBoost Model C baseline.
Does NOT modify any existing model files.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from src.data.loader import load_dataset
from src.grading.fuzzy_grader import FuzzyGrader
from src.grading.feature_engineer import FeatureEngineer
from src.symbolic.pipeline import SymbolicPipeline
from src.models.base import get_xgb_params_c, RANDOM_STATE, N_SPLITS, DISEASES


def build_X(X_clinical: pd.DataFrame) -> pd.DataFrame:
    grader = FuzzyGrader()
    X_fuzzy = grader.grade(X_clinical).reset_index(drop=True)
    engineer = FeatureEngineer()
    X_eng = engineer.engineer(X_fuzzy)
    pipeline = SymbolicPipeline("rules")
    X_sym = pipeline.transform(X_fuzzy).reset_index(drop=True)
    return pd.concat([X_fuzzy, X_eng, X_sym], axis=1)


def cv_eval(model_fn, X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s, train_accs = [], [], []
    all_true, all_pred = [], []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_fn()
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        if hasattr(val_pred, 'flatten'):
            val_pred = val_pred.flatten()
        if hasattr(train_pred, 'flatten'):
            train_pred = train_pred.flatten()

        train_accs.append(accuracy_score(y_train, train_pred))
        accs.append(accuracy_score(y_val, val_pred))
        f1s.append(f1_score(y_val, val_pred, average="macro", zero_division=0))
        all_true.extend(y_val.tolist())
        all_pred.extend(val_pred.tolist())

    per_class = f1_score(all_true, all_pred, average=None, zero_division=0)
    return {
        "label": label,
        "train_acc": float(np.mean(train_accs)),
        "val_acc": float(np.mean(accs)),
        "val_std": float(np.std(accs)),
        "val_f1": float(np.mean(f1s)),
        "gap": float(np.mean(train_accs)) - float(np.mean(accs)),
        "per_class_f1": {DISEASES[i]: round(float(per_class[i]), 4) for i in range(len(DISEASES))},
    }


X_clinical, _, _, y = load_dataset()
X = build_X(X_clinical)
y = y.reset_index(drop=True)

print(f"Feature matrix: {X.shape[0]} patients x {X.shape[1]} features\n")

# XGBoost (current Model C baseline)
def xgb_fn():
    return XGBClassifier(**get_xgb_params_c(), verbosity=0)

# CatBoost — equivalent regularization strength
def cat_fn():
    return CatBoostClassifier(
        iterations=300,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=10.0,
        bootstrap_type="Bernoulli",
        subsample=0.7,
        random_seed=RANDOM_STATE,
        verbose=0,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        od_type="Iter",
        od_wait=30,
    )

results = [
    cv_eval(xgb_fn, X, y, "XGBoost (current Model C)"),
    cv_eval(cat_fn, X, y, "CatBoost"),
]

print("=" * 80)
print(f"{'Model':<28} {'Train':>8} {'Val Acc':>10} {'Std':>8} {'Gap':>8} {'F1':>8}")
print("-" * 80)
for r in results:
    print(f"{r['label']:<28} {r['train_acc']:>8.4f} {r['val_acc']:>10.4f} "
          f"{r['val_std']:>8.4f} {r['gap']:>+8.4f} {r['val_f1']:>8.4f}")
print("=" * 80)

print("\nPer-class F1:")
print(f"{'Disease':<30} {'XGBoost':>10} {'CatBoost':>10} {'Delta':>8}")
print("-" * 60)
for d in DISEASES:
    xgb_f1 = results[0]["per_class_f1"][d]
    cat_f1 = results[1]["per_class_f1"][d]
    delta = cat_f1 - xgb_f1
    marker = " +" if delta > 0.01 else (" -" if delta < -0.01 else "  ")
    print(f"{d:<30} {xgb_f1:>10.4f} {cat_f1:>10.4f} {delta:>+8.4f}{marker}")

print("\nVerdict:")
acc_delta = results[1]["val_acc"] - results[0]["val_acc"]
gap_delta = results[1]["gap"] - results[0]["gap"]
print(f"  Accuracy change:   {acc_delta:+.4f} ({acc_delta*100:+.2f}pp)")
print(f"  Overfitting gap:   XGBoost {results[0]['gap']:+.4f}  CatBoost {results[1]['gap']:+.4f}")
if acc_delta > 0.005 and results[1]["gap"] <= results[0]["gap"] + 0.02:
    print("  RESULT: CatBoost BETTER — worth replacing XGBoost in Model C")
elif acc_delta > 0 and results[1]["gap"] > results[0]["gap"] + 0.03:
    print("  RESULT: CatBoost marginally better accuracy but MORE overfitting — not worth it")
else:
    print("  RESULT: No meaningful improvement — keep XGBoost")
