"""
Expanded CatBoost sweep for Model C — finding #8 in changes-made.md.
Holds the existing best 5 params fixed (iterations=200, depth=3, l2_leaf_reg=5.0,
learning_rate=0.05, subsample=0.8) and sweeps 3 additional regularization axes
that were silently left at CatBoost defaults during the XGBoost migration:
  - rsm               (feature subsampling; old XGBoost had colsample_bytree=0.4)
  - min_data_in_leaf  (a leaf can currently form around a single patient)
  - random_strength   (split-scoring randomness, orthogonal to the 5 already swept)

Cheaper than a full new cross-product grid — tests whether these 3 axes help
without re-deriving the existing 5. No od_type/od_wait: removed in finding #9,
inert without eval_set and costs accuracy if wired up on this small dataset.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from itertools import product

from src.data.loader import load_dataset
from src.grading.fuzzy_grader import FuzzyGrader
from src.grading.feature_engineer import FeatureEngineer
from src.symbolic.pipeline import SymbolicPipeline
from src.models.base import RANDOM_STATE, N_SPLITS

X_clinical, _, _, y = load_dataset()
grader = FuzzyGrader()
X_fuzzy = grader.grade(X_clinical).reset_index(drop=True)
engineer = FeatureEngineer()
X_eng = engineer.engineer(X_fuzzy)
pipeline = SymbolicPipeline("rules")
X_sym = pipeline.transform(X_fuzzy).reset_index(drop=True)
X = pd.concat([X_fuzzy, X_eng, X_sym], axis=1)
y = y.reset_index(drop=True)

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

FIXED = dict(
    iterations=200, depth=3, l2_leaf_reg=5.0, learning_rate=0.05, subsample=0.8,
    bootstrap_type="Bernoulli", random_seed=RANDOM_STATE, verbose=0,
    loss_function="MultiClass",
)

grid = {
    "rsm":              [1.0, 0.8, 0.6, 0.4],
    "min_data_in_leaf": [1, 3, 5],
    "random_strength":  [1, 2, 5],
}

combos = list(product(*grid.values()))
keys = list(grid.keys())
print(f"Testing {len(combos)} combinations (baseline: rsm=1.0, min_data_in_leaf=1, random_strength=1)...\n")

results = []
for i, vals in enumerate(combos):
    params = {**FIXED, **dict(zip(keys, vals))}
    accs, train_accs, f1s = [], [], []
    for train_idx, val_idx in cv.split(X, y):
        model = CatBoostClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        train_pred = model.predict(X.iloc[train_idx]).flatten()
        val_pred = model.predict(X.iloc[val_idx]).flatten()
        train_accs.append(accuracy_score(y.iloc[train_idx], train_pred))
        accs.append(accuracy_score(y.iloc[val_idx], val_pred))
        f1s.append(f1_score(y.iloc[val_idx], val_pred, average="macro", zero_division=0))
    val_acc = float(np.mean(accs))
    val_f1 = float(np.mean(f1s))
    gap = float(np.mean(train_accs)) - val_acc
    results.append({**dict(zip(keys, vals)), "val_acc": val_acc, "val_std": float(np.std(accs)),
                     "val_f1": val_f1, "gap": gap})
    if (i + 1) % 9 == 0:
        print(f"  {i+1}/{len(combos)} done...")

by_acc = sorted(results, key=lambda r: r["val_acc"], reverse=True)
by_f1 = sorted(results, key=lambda r: r["val_f1"], reverse=True)

print("\nTOP 10 by accuracy:")
print(f"{'rsm':>5} {'mdl':>5} {'rs':>4} {'ValAcc':>8} {'Std':>7} {'F1':>8} {'Gap':>8}")
print("-" * 55)
for r in by_acc[:10]:
    print(f"{r['rsm']:>5} {r['min_data_in_leaf']:>5} {r['random_strength']:>4} "
          f"{r['val_acc']:>8.4f} {r['val_std']:>7.4f} {r['val_f1']:>8.4f} {r['gap']:>+8.4f}")

print("\nTOP 10 by macro F1:")
print(f"{'rsm':>5} {'mdl':>5} {'rs':>4} {'ValAcc':>8} {'Std':>7} {'F1':>8} {'Gap':>8}")
print("-" * 55)
for r in by_f1[:10]:
    print(f"{r['rsm']:>5} {r['min_data_in_leaf']:>5} {r['random_strength']:>4} "
          f"{r['val_acc']:>8.4f} {r['val_std']:>7.4f} {r['val_f1']:>8.4f} {r['gap']:>+8.4f}")

baseline = next(r for r in results if r["rsm"] == 1.0 and r["min_data_in_leaf"] == 1 and r["random_strength"] == 1)
best = by_acc[0]
print(f"\nCurrent baseline (rsm=1.0, mdl=1, rs=1): ValAcc={baseline['val_acc']:.4f}  F1={baseline['val_f1']:.4f}  Gap={baseline['gap']:+.4f}")
print(f"Best found:                              ValAcc={best['val_acc']:.4f}  F1={best['val_f1']:.4f}  Gap={best['gap']:+.4f}  "
      f"(rsm={best['rsm']}, mdl={best['min_data_in_leaf']}, rs={best['random_strength']})")
delta = best['val_acc'] - baseline['val_acc']
print(f"Delta vs baseline: {delta:+.4f} ({delta*100:+.2f}pp)")
