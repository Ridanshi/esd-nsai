"""
Empirical calibration of the four rule-tier baseline multipliers (A/B/C/D),
evaluated under the project's standard stratified 10-fold CV protocol
(N_SPLITS=10, RANDOM_STATE=42, matching src/models/base.py).

Baseline: tier multipliers fixed at (1.0, 1.0, 1.0, 1.0) -- i.e. each rule's
hand-picked weight (as authored in rules/*.yaml) used as-is, unscaled.

Calibrated: per-fold, a bounded search finds a global per-tier multiplier
(m_A, m_B, m_C, m_D) that maximises macro-F1 of the rule engine's own
argmax(certainty) prediction on the training fold, evaluated on the held-out
fold. Bounds are set around the tier's semantic role so pathognomonic (A)
cannot be optimised below supportive (B), etc. -- the four-tier *ordering*
is preserved by construction, only the numeric weight within each tier's
role is data-calibrated.

This targets the rule engine's own discriminative quality (the object the
patent's "Four-Tier Symbolic Rule Engine" claim is about), independent of
the downstream CatBoost classifier.
"""
import sys
import numpy as np
import pandas as pd
import yaml
import glob
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.dirname(__file__))
from src.data.loader import load_dataset
from src.grading.fuzzy_grader import FuzzyGrader

DISEASES = [
    "psoriasis", "seborrheic_dermatitis", "lichen_planus",
    "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris"
]
DISEASE_IDX = {d: i for i, d in enumerate(DISEASES)}
TIERS = ["A", "B", "C", "D"]
TIER_IDX = {t: i for i, t in enumerate(TIERS)}

N_SPLITS = 10
RANDOM_STATE = 42

# Bounds keep each tier within its intended semantic role relative to the
# others (pathognomonic > supportive > auxiliary), centred on the original
# hand-picked baseline (1.0, 0.6, 0.3, 0.4) expressed here as multipliers
# on each rule's own already-tiered weight.
BOUNDS = [
    (0.85, 1.30),   # m_A
    (0.55, 1.00),   # m_B
    (0.30, 0.80),   # m_C
    (0.30, 1.00),   # m_D
]


def load_rules(rules_dir):
    rules = []
    for path in glob.glob(os.path.join(rules_dir, "*.yaml")):
        with open(path) as f:
            rules.extend(yaml.safe_load(f))
    return rules


def build_strength_matrix(X_fuzzy: pd.DataFrame, rules: list):
    """(n_patients, n_rules) firing strength -- independent of weights/multipliers."""
    n = len(X_fuzzy)
    n_rules = len(rules)
    strength = np.zeros((n, n_rules), dtype=float)
    disease_of = np.zeros(n_rules, dtype=int)
    tier_of = np.zeros(n_rules, dtype=int)
    weight_of = np.zeros(n_rules, dtype=float)

    Xv = X_fuzzy

    for j, rule in enumerate(rules):
        disease_of[j] = DISEASE_IDX[rule["disease"]]
        tier_of[j] = TIER_IDX[rule.get("tier")]
        weight_of[j] = rule["weight"]

        conds = rule["conditions"]
        vals = np.stack([Xv[c["feature"]].to_numpy(dtype=float) for c in conds], axis=1)
        thresholds = np.array([c["threshold"] for c in conds], dtype=float)
        gate = np.all(vals >= thresholds, axis=1)
        min_strength = vals.min(axis=1)
        strength[:, j] = np.where(gate, min_strength, 0.0)

    return strength, disease_of, tier_of, weight_of


def score_patients(strength, disease_of, tier_of, weight_of, multipliers):
    """Vectorised re-implementation of RuleEngine.fire() with tier multipliers."""
    m = np.asarray(multipliers, dtype=float)
    eff_weight = weight_of * m[tier_of]                       # (n_rules,)
    contribution = strength * eff_weight[None, :]              # (n_patients, n_rules)

    n = strength.shape[0]
    n_diseases = len(DISEASES)
    accumulated = np.zeros((n, n_diseases))
    penalty = np.zeros((n, n_diseases))
    max_w = np.zeros(n_diseases)

    is_d = (tier_of == TIER_IDX["D"])
    for d_idx in range(n_diseases):
        rule_mask = (disease_of == d_idx)
        non_d_mask = rule_mask & ~is_d
        d_mask = rule_mask & is_d

        accumulated[:, d_idx] = contribution[:, non_d_mask].sum(axis=1)
        penalty[:, d_idx] = contribution[:, d_mask].sum(axis=1)
        max_w[d_idx] = eff_weight[non_d_mask].sum()

    max_w_safe = np.where(max_w == 0.0, 1.0, max_w)
    raw = (accumulated - penalty) / max_w_safe[None, :]
    scores = np.clip(raw, 0.0, 1.0)
    scores[:, max_w == 0.0] = 0.0
    return scores


def macro_f1_for_multipliers(multipliers, strength, disease_of, tier_of, weight_of, y):
    scores = score_patients(strength, disease_of, tier_of, weight_of, multipliers)
    y_pred = scores.argmax(axis=1)
    return f1_score(y, y_pred, average="macro", zero_division=0)


def main():
    print("Loading dataset...")
    X_clinical, _, _, y = load_dataset()
    y = y.to_numpy()

    grader = FuzzyGrader()
    X_fuzzy = grader.grade(X_clinical)

    rules_dir = os.path.join(os.path.dirname(__file__), "rules")
    rules = load_rules(rules_dir)
    print(f"Loaded {len(rules)} rules.")

    strength, disease_of, tier_of, weight_of = build_strength_matrix(X_fuzzy, rules)

    baseline_m = (1.0, 1.0, 1.0, 1.0)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    baseline_accs, baseline_f1s = [], []
    calib_accs, calib_f1s = [], []
    fold_multipliers = []

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(strength, y)):
        s_tr, s_val = strength[train_idx], strength[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Baseline, evaluated on this fold's held-out set (no fitting -- fixed weights)
        base_scores_val = score_patients(s_val, disease_of, tier_of, weight_of, baseline_m)
        base_pred_val = base_scores_val.argmax(axis=1)
        baseline_accs.append(accuracy_score(y_val, base_pred_val))
        baseline_f1s.append(f1_score(y_val, base_pred_val, average="macro", zero_division=0))

        # Calibrated: optimise multipliers on train fold only
        def objective(m):
            return -macro_f1_for_multipliers(m, s_tr, disease_of, tier_of, weight_of, y_tr)

        result = differential_evolution(
            objective, BOUNDS, seed=RANDOM_STATE, maxiter=60, popsize=15,
            tol=1e-6, polish=True, workers=1, updating="deferred"
        )
        best_m = result.x
        fold_multipliers.append(best_m)

        calib_scores_val = score_patients(s_val, disease_of, tier_of, weight_of, best_m)
        calib_pred_val = calib_scores_val.argmax(axis=1)
        calib_accs.append(accuracy_score(y_val, calib_pred_val))
        calib_f1s.append(f1_score(y_val, calib_pred_val, average="macro", zero_division=0))

        print(f"Fold {fold_i+1:2d}/10 | baseline acc={baseline_accs[-1]:.4f} f1={baseline_f1s[-1]:.4f} "
              f"| calibrated acc={calib_accs[-1]:.4f} f1={calib_f1s[-1]:.4f} | m={np.round(best_m,3)}")

    print()
    print("=" * 70)
    print(f"BASELINE   (1.0,1.0,1.0,1.0 -- rules used as hand-authored, unscaled):")
    print(f"  accuracy: {np.mean(baseline_accs):.4f} +/- {np.std(baseline_accs):.4f}")
    print(f"  macro-F1: {np.mean(baseline_f1s):.4f} +/- {np.std(baseline_f1s):.4f}")
    print()
    print(f"CALIBRATED (per-fold optimised tier multipliers, held-out eval):")
    print(f"  accuracy: {np.mean(calib_accs):.4f} +/- {np.std(calib_accs):.4f}")
    print(f"  macro-F1: {np.mean(calib_f1s):.4f} +/- {np.std(calib_f1s):.4f}")
    print()
    mean_m = np.mean(fold_multipliers, axis=0)
    print(f"Mean calibrated multipliers across folds (m_A, m_B, m_C, m_D): {np.round(mean_m, 4)}")
    print(f"Per-fold multipliers:")
    for i, m in enumerate(fold_multipliers):
        print(f"  fold {i+1:2d}: {np.round(m, 4)}")

    # Final: fit once on the full 366-patient dataset for the deployable multiplier set
    def objective_full(m):
        return -macro_f1_for_multipliers(m, strength, disease_of, tier_of, weight_of, y)

    result_full = differential_evolution(
        objective_full, BOUNDS, seed=RANDOM_STATE, maxiter=100, popsize=20,
        tol=1e-7, polish=True, workers=1, updating="deferred"
    )
    print()
    print(f"Full-dataset calibrated multipliers: {np.round(result_full.x, 4)}  "
          f"(macro-F1 on full set: {-result_full.fun:.4f})")
    base_scores_full = score_patients(strength, disease_of, tier_of, weight_of, baseline_m)
    base_pred_full = base_scores_full.argmax(axis=1)
    print(f"Full-dataset baseline macro-F1: {f1_score(y, base_pred_full, average='macro', zero_division=0):.4f}, "
          f"accuracy: {accuracy_score(y, base_pred_full):.4f}")
    calib_scores_full = score_patients(strength, disease_of, tier_of, weight_of, result_full.x)
    calib_pred_full = calib_scores_full.argmax(axis=1)
    print(f"Full-dataset calibrated accuracy: {accuracy_score(y, calib_pred_full):.4f}")


if __name__ == "__main__":
    main()
