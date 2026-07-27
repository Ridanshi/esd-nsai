"""
Rule-weight ablation study — finding #6 in changes-made.md.

Measurement tool only — does NOT modify rules/*.yaml. Ridanshi's textbook-sourced
weights are read-only evidence here; this produces the outcome-impact numbers the
rest of the pipeline already has (MI scoring for engineered features, CV sweep for
hyperparameters) but the rule library has never had.

Method: for each of the 45 rules, in memory, zero out its weight AND its share of
that disease's max_weight normalizer (i.e. simulate "this rule doesn't exist" for
that disease's certainty scoring — not just "this rule always fails to fire", which
would double-penalize by leaving a stale normalizer). Rebuild the full 29-feature
pipeline, rerun identical 10-fold CV CatBoost, compare to baseline. Restore the
weight before testing the next rule. ~46 CV runs (1 baseline + 45 ablations) x 200
trees x 10 folds.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from src.data.loader import load_dataset
from src.grading.fuzzy_grader import FuzzyGrader
from src.grading.feature_engineer import FeatureEngineer
from src.symbolic.pipeline import SymbolicPipeline
from src.models.base import get_catboost_params_c, RANDOM_STATE, N_SPLITS


def cv_accuracy(X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for tr, va in cv.split(X, y):
        model = CatBoostClassifier(**get_catboost_params_c())
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va]).flatten()
        accs.append(accuracy_score(y.iloc[va], pred))
        f1s.append(f1_score(y.iloc[va], pred, average="macro", zero_division=0))
    return float(np.mean(accs)), float(np.mean(f1s))


X_clinical, _, _, y = load_dataset()
X_fuzzy = FuzzyGrader().grade(X_clinical).reset_index(drop=True)
X_eng = FeatureEngineer().engineer(X_fuzzy)
y = y.reset_index(drop=True)

pipeline = SymbolicPipeline("rules")
engine = pipeline._rule_engine

X_sym_baseline = pipeline.transform(X_fuzzy)
X_baseline = pd.concat([X_fuzzy, X_eng, X_sym_baseline], axis=1)
baseline_acc, baseline_f1 = cv_accuracy(X_baseline, y)
print(f"Baseline (all 45 rules): acc={baseline_acc:.4f}  macroF1={baseline_f1:.4f}\n")

n_rules = len(engine._rules)
print(f"Ablating {n_rules} rules one at a time...\n")

results = []
for i, rule in enumerate(engine._rules):
    rule_id = rule["id"]
    disease = rule["disease"]
    tier = rule.get("tier")
    weight = rule["weight"]

    rule["weight"] = 0.0
    if tier != "D":
        engine._max_weight[disease] -= weight

    X_sym = pipeline.transform(X_fuzzy)
    X = pd.concat([X_fuzzy, X_eng, X_sym], axis=1)
    acc, f1 = cv_accuracy(X, y)

    rule["weight"] = weight
    if tier != "D":
        engine._max_weight[disease] += weight

    delta_acc = baseline_acc - acc
    delta_f1 = baseline_f1 - f1
    results.append({
        "id": rule_id, "disease": disease, "tier": tier, "weight": weight,
        "acc": acc, "delta_acc": delta_acc, "f1": f1, "delta_f1": delta_f1,
    })
    if (i + 1) % 9 == 0:
        print(f"  {i+1}/{n_rules} done...")

results.sort(key=lambda r: r["delta_acc"], reverse=True)

print(f"\n{'Rule':<10} {'Disease':<26} {'Tier':>4} {'Weight':>7} {'AccΔ':>9} {'F1Δ':>9}  Interpretation")
print("-" * 100)
for r in results:
    if r["delta_acc"] > 0.005:
        interp = "removing HURTS — rule pulls real weight"
    elif r["delta_acc"] < -0.005:
        interp = "removing HELPS — rule may be net-negative"
    else:
        interp = "no measurable effect"
    print(f"{r['id']:<10} {r['disease']:<26} {r['tier'] or '-':>4} {r['weight']:>7.2f} "
          f"{r['delta_acc']:>+9.4f} {r['delta_f1']:>+9.4f}  {interp}")

n_matter = sum(1 for r in results if abs(r["delta_acc"]) > 0.005)
print(f"\n{n_matter}/{n_rules} rules show measurable CV accuracy impact (|Δacc| > 0.5pp) when individually ablated.")
print("No weights changed in rules/*.yaml — this is measurement only.")
