"""
Diagnose Model C fit: compare train vs validation accuracy across 10 folds.
Train >> Val  → overfitting (regularize more)
Train ≈ Val, both low → underfitting (relax regularization)
Train ≈ Val, both high → good fit
"""
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from src.data.loader import load_dataset
from src.grading.fuzzy_grader import FuzzyGrader
from src.grading.feature_engineer import FeatureEngineer
from src.symbolic.pipeline import SymbolicPipeline
from src.models.base import get_xgb_params_c, RANDOM_STATE, N_SPLITS
import pandas as pd

X_clinical, _, _, y = load_dataset()
grader = FuzzyGrader()
X_fuzzy = grader.grade(X_clinical).reset_index(drop=True)
engineer = FeatureEngineer()
X_engineered = engineer.engineer(X_fuzzy)
pipeline = SymbolicPipeline("rules")
X_symbolic = pipeline.transform(X_fuzzy).reset_index(drop=True)
X = pd.concat([X_fuzzy, X_engineered, X_symbolic], axis=1)

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

train_accs, val_accs = [], []
train_f1s, val_f1s = [], []

print(f"{'Fold':<6} {'Train Acc':>10} {'Val Acc':>10} {'Gap':>8} {'Train F1':>10} {'Val F1':>10}")
print("-" * 58)

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(**get_xgb_params_c())
    model.fit(X_train, y_train, verbose=False)

    train_pred = model.predict(X_train)
    val_pred   = model.predict(X_val)

    t_acc = accuracy_score(y_train, train_pred)
    v_acc = accuracy_score(y_val, val_pred)
    t_f1  = f1_score(y_train, train_pred, average="macro", zero_division=0)
    v_f1  = f1_score(y_val,   val_pred,   average="macro", zero_division=0)

    train_accs.append(t_acc)
    val_accs.append(v_acc)
    train_f1s.append(t_f1)
    val_f1s.append(v_f1)

    gap = t_acc - v_acc
    print(f"{fold:<6} {t_acc:>10.4f} {v_acc:>10.4f} {gap:>+8.4f} {t_f1:>10.4f} {v_f1:>10.4f}")

print("-" * 58)
mean_gap = np.mean(train_accs) - np.mean(val_accs)
print(f"{'Mean':<6} {np.mean(train_accs):>10.4f} {np.mean(val_accs):>10.4f} {mean_gap:>+8.4f} "
      f"{np.mean(train_f1s):>10.4f} {np.mean(val_f1s):>10.4f}")
print(f"{'Std':<6} {np.std(train_accs):>10.4f} {np.std(val_accs):>10.4f}")

print()
if mean_gap > 0.08:
    print("VERDICT: OVERFITTING — train much higher than val. Regularize more.")
elif np.mean(val_accs) < 0.82 and mean_gap < 0.04:
    print("VERDICT: UNDERFITTING — both train and val low. Relax regularization.")
elif mean_gap < 0.04:
    print("VERDICT: GOOD FIT — train ≈ val. Accuracy ceiling is the dataset limit.")
else:
    print("VERDICT: MILD OVERFIT — small gap. Try slight regularization increase.")
