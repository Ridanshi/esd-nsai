"""
CatBoost hyperparameter sweep for Model C (29 features, 366 samples).
Grid: iterations, depth, l2_leaf_reg, learning_rate, subsample.
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

# Build 29-feature matrix
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

grid = {
    "iterations":    [200, 300, 500],
    "depth":         [3, 4, 6],
    "l2_leaf_reg":   [5, 10, 20],
    "learning_rate": [0.03, 0.05],
    "subsample":     [0.7, 0.8],
}

combos = list(product(*grid.values()))
keys = list(grid.keys())
print(f"Testing {len(combos)} combinations...\n")

results = []
for i, vals in enumerate(combos):
    params = dict(zip(keys, vals))
    accs, train_accs = [], []

    for train_idx, val_idx in cv.split(X, y):
        model = CatBoostClassifier(
            **params,
            bootstrap_type="Bernoulli",
            random_seed=RANDOM_STATE,
            verbose=0,
            loss_function="MultiClass",
            od_type="Iter",
            od_wait=30,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        train_accs.append(accuracy_score(y.iloc[train_idx], model.predict(X.iloc[train_idx]).flatten()))
        accs.append(accuracy_score(y.iloc[val_idx], model.predict(X.iloc[val_idx]).flatten()))

    val_acc = float(np.mean(accs))
    gap = float(np.mean(train_accs)) - val_acc
    results.append({**params, "val_acc": val_acc, "val_std": float(np.std(accs)), "gap": gap})

    if (i + 1) % 18 == 0:
        print(f"  {i+1}/{len(combos)} done...")

results.sort(key=lambda r: r["val_acc"], reverse=True)

print("\nTOP 10 configurations:")
print(f"{'iter':>5} {'dep':>4} {'l2':>5} {'lr':>6} {'sub':>5} {'ValAcc':>8} {'Std':>7} {'Gap':>7}")
print("-" * 60)
for r in results[:10]:
    print(f"{r['iterations']:>5} {r['depth']:>4} {r['l2_leaf_reg']:>5} "
          f"{r['learning_rate']:>6.3f} {r['subsample']:>5.1f} "
          f"{r['val_acc']:>8.4f} {r['val_std']:>7.4f} {r['gap']:>+7.4f}")

best = results[0]
print(f"\nWINNER: iterations={best['iterations']} depth={best['depth']} "
      f"l2={best['l2_leaf_reg']} lr={best['learning_rate']} sub={best['subsample']}")
print(f"  ValAcc={best['val_acc']:.4f}  Std={best['val_std']:.4f}  Gap={best['gap']:+.4f}")
print(f"\nCurrent baseline: ValAcc=0.8716  Std=0.0325  Gap=+0.0388")
delta = best['val_acc'] - 0.8716
print(f"Delta vs current: {delta:+.4f} ({delta*100:+.2f}pp)")
