"""
Feature pruning via RFECV — finds optimal subset of the 29 features.
Uses CatBoost with current best params. Reports accuracy, gap, and which features were dropped.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from src.data.loader import load_dataset
from src.grading.fuzzy_grader import FuzzyGrader
from src.grading.feature_engineer import FeatureEngineer
from src.symbolic.pipeline import SymbolicPipeline
from src.models.base import get_catboost_params_c, RANDOM_STATE, N_SPLITS, DISEASES


def build_X(X_clinical):
    grader = FuzzyGrader()
    X_fuzzy = grader.grade(X_clinical).reset_index(drop=True)
    X_eng = FeatureEngineer().engineer(X_fuzzy)
    X_sym = SymbolicPipeline("rules").transform(X_fuzzy).reset_index(drop=True)
    return pd.concat([X_fuzzy, X_eng, X_sym], axis=1)


def cv_eval(X, y, label):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accs, train_accs = [], []
    all_true, all_pred = [], []
    for train_idx, val_idx in cv.split(X, y):
        model = CatBoostClassifier(**get_catboost_params_c())
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        train_pred = model.predict(X.iloc[train_idx]).flatten()
        val_pred = model.predict(X.iloc[val_idx]).flatten()
        train_accs.append(accuracy_score(y.iloc[train_idx], train_pred))
        accs.append(accuracy_score(y.iloc[val_idx], val_pred))
        all_true.extend(y.iloc[val_idx].tolist())
        all_pred.extend(val_pred.tolist())
    per_class = f1_score(all_true, all_pred, average=None, zero_division=0)
    return {
        "label": label,
        "n_features": X.shape[1],
        "train_acc": float(np.mean(train_accs)),
        "val_acc": float(np.mean(accs)),
        "val_std": float(np.std(accs)),
        "val_f1": float(np.mean([f1_score(all_true, all_pred, average="macro", zero_division=0)])),
        "gap": float(np.mean(train_accs)) - float(np.mean(accs)),
        "per_class_f1": {DISEASES[i]: round(float(per_class[i]), 4) for i in range(len(DISEASES))},
        "macro_f1": float(f1_score(all_true, all_pred, average="macro", zero_division=0)),
    }


X_clinical, _, _, y = load_dataset()
X = build_X(X_clinical)
y = y.reset_index(drop=True)

print(f"Starting features: {X.shape[1]}")
print("Running RFECV (this takes a few minutes)...\n")

# Use a lightweight CatBoost for RFECV (faster)
selector_model = CatBoostClassifier(
    iterations=100, depth=3, learning_rate=0.05,
    l2_leaf_reg=5.0, bootstrap_type="Bernoulli", subsample=0.8,
    random_seed=RANDOM_STATE, verbose=0, loss_function="MultiClass",
)

rfecv = RFECV(
    estimator=selector_model,
    step=1,
    cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
    scoring="accuracy",
    min_features_to_select=10,
    n_jobs=1,
)
rfecv.fit(X, y)

selected = X.columns[rfecv.support_].tolist()
dropped = X.columns[~rfecv.support_].tolist()
X_pruned = X[selected]

print(f"Optimal features: {rfecv.n_features_} / {X.shape[1]}")
print(f"Dropped ({len(dropped)}): {dropped}")
print(f"Kept   ({len(selected)}): {selected}\n")

# Full eval: baseline (29) vs pruned
baseline = cv_eval(X, y, f"All 29 features (baseline)")
pruned = cv_eval(X_pruned, y, f"Pruned {rfecv.n_features_} features (RFECV)")

print("=" * 75)
print(f"{'Config':<35} {'Features':>8} {'ValAcc':>8} {'Std':>7} {'Gap':>7} {'F1':>8}")
print("-" * 75)
for r in [baseline, pruned]:
    print(f"{r['label']:<35} {r['n_features']:>8} {r['val_acc']:>8.4f} "
          f"{r['val_std']:>7.4f} {r['gap']:>+7.4f} {r['macro_f1']:>8.4f}")
print("=" * 75)

delta_acc = pruned['val_acc'] - baseline['val_acc']
delta_gap = pruned['gap'] - baseline['gap']
print(f"\nDelta accuracy: {delta_acc:+.4f} ({delta_acc*100:+.2f}pp)")
print(f"Delta gap:      {delta_gap:+.4f}")

print("\nPer-class F1:")
print(f"{'Disease':<30} {'Baseline':>10} {'Pruned':>10} {'Delta':>8}")
print("-" * 60)
for d in DISEASES:
    b = baseline["per_class_f1"][d]
    p = pruned["per_class_f1"][d]
    print(f"{d:<30} {b:>10.4f} {p:>10.4f} {p-b:>+8.4f}")

print("\nVerdict:")
if pruned['val_acc'] > baseline['val_acc'] and pruned['gap'] <= baseline['gap'] + 0.02:
    print("  INTEGRATE — pruned model is better with no overfitting increase")
elif pruned['val_acc'] > baseline['val_acc'] and pruned['gap'] > baseline['gap'] + 0.02:
    print("  SKIP — accuracy gain but overfitting increased")
else:
    print("  SKIP — no accuracy gain from pruning")
