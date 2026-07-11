"""
Regularization sweep for Model C. Goal: shrink train-val gap below 5%
while maximising val accuracy.
"""
import numpy as np
import pandas as pd
from itertools import product
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from src.data.loader import load_dataset
from src.grading.fuzzy_grader import FuzzyGrader
from src.grading.feature_engineer import FeatureEngineer
from src.symbolic.pipeline import SymbolicPipeline
from src.models.base import RANDOM_STATE, N_SPLITS

X_clinical, _, _, y = load_dataset()
grader = FuzzyGrader()
X_fuzzy = grader.grade(X_clinical).reset_index(drop=True)
engineer = FeatureEngineer()
X_engineered = engineer.engineer(X_fuzzy)
pipeline = SymbolicPipeline("rules")
X_symbolic = pipeline.transform(X_fuzzy).reset_index(drop=True)
X = pd.concat([X_fuzzy, X_engineered, X_symbolic], axis=1)

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

BASE = dict(max_depth=3, learning_rate=0.05, eval_metric="mlogloss", random_state=RANDOM_STATE)

grid = {
    "n_estimators":    [100, 150, 200],
    "reg_lambda":      [5.0, 10.0, 20.0],
    "subsample":       [0.6, 0.7],
    "colsample_bytree":[0.4, 0.5],
    "min_child_weight":[3, 5],
}

keys = list(grid.keys())
combos = list(product(*[grid[k] for k in keys]))

print(f"Testing {len(combos)} combinations...\n")
print(f"{'#':<4} {'n_est':>5} {'lam':>5} {'sub':>4} {'col':>4} {'mcw':>4} "
      f"{'ValAcc':>7} {'ValF1':>6} {'Gap':>7}")
print("-" * 60)

results = []
for i, combo in enumerate(combos, 1):
    params = {**BASE, **dict(zip(keys, combo))}
    train_accs, val_accs, val_f1s = [], [], []
    for train_idx, val_idx in cv.split(X, y):
        model = XGBClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)
        val_pred   = model.predict(X.iloc[val_idx])
        train_pred = model.predict(X.iloc[train_idx])
        val_accs.append(accuracy_score(y.iloc[val_idx], val_pred))
        train_accs.append(accuracy_score(y.iloc[train_idx], train_pred))
        val_f1s.append(f1_score(y.iloc[val_idx], val_pred, average="macro", zero_division=0))

    v = np.mean(val_accs)
    t = np.mean(train_accs)
    f = np.mean(val_f1s)
    gap = t - v
    results.append((v, f, gap, combo))
    n, lam, sub, col, mcw = combo
    print(f"{i:<4} {n:>5} {lam:>5} {sub:>4} {col:>4} {mcw:>4} "
          f"{v:>7.4f} {f:>6.4f} {gap:>+7.4f}")

print("\n--- TOP 5 by Val Acc (gap < 0.08) ---")
filtered = [(v, f, g, c) for v, f, g, c in results if g < 0.08]
filtered.sort(key=lambda x: -x[0])
for v, f, g, c in filtered[:5]:
    n, lam, sub, col, mcw = c
    print(f"ValAcc={v:.4f} ValF1={f:.4f} Gap={g:+.4f} | "
          f"n_est={n} lam={lam} sub={sub} col={col} mcw={mcw}")

if not filtered:
    print("No combo achieved gap < 0.08. Best by val acc:")
    results.sort(key=lambda x: -x[0])
    for v, f, g, c in results[:5]:
        n, lam, sub, col, mcw = c
        print(f"ValAcc={v:.4f} Gap={g:+.4f} | n_est={n} lam={lam} sub={sub} col={col} mcw={mcw}")
