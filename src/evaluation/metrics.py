import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from src.models.base import get_xgb_params, DISEASES, RANDOM_STATE


def print_comparison_table(results_a: dict, results_b: dict, results_c: dict) -> None:
    header = f"{'Model':<40} {'Accuracy':>16} {'Macro F1':>16}"
    print("\n" + "=" * 72)
    print(header)
    print("-" * 72)
    for res in [results_a, results_b, results_c]:
        label = res["label"]
        acc = f"{res['accuracy_mean']:.4f} +/- {res['accuracy_std']:.4f}"
        f1 = f"{res['macro_f1_mean']:.4f} +/- {res['macro_f1_std']:.4f}"
        print(f"{label:<40} {acc:>16} {f1:>16}")
    print("=" * 72)

    print("\nPer-class F1 scores:")
    print(f"{'Disease':<35} {'Model A':>10} {'Model B':>10} {'Model C':>10}")
    print("-" * 65)
    for disease in DISEASES:
        fa = results_a["per_class_f1"].get(disease, 0.0)
        fb = results_b["per_class_f1"].get(disease, 0.0)
        fc = results_c["per_class_f1"].get(disease, 0.0)
        print(f"{disease:<35} {fa:>10.4f} {fb:>10.4f} {fc:>10.4f}")


def run_statistical_test(
    X_b: pd.DataFrame,
    X_c: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """
    5x2 paired cross-validation t-test comparing Model B vs Model C.
    Method: Dietterich (1998) — standard for comparing two classifiers on small datasets.
    """
    differences = []
    for i in range(5):
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE + i)
        fold_diffs = []
        for train_idx, val_idx in cv.split(X_b, y):
            clf_b = XGBClassifier(**get_xgb_params())
            clf_b.fit(X_b.iloc[train_idx], y.iloc[train_idx], verbose=False)
            acc_b = accuracy_score(y.iloc[val_idx], clf_b.predict(X_b.iloc[val_idx]))

            clf_c = XGBClassifier(**get_xgb_params())
            clf_c.fit(X_c.iloc[train_idx], y.iloc[train_idx], verbose=False)
            acc_c = accuracy_score(y.iloc[val_idx], clf_c.predict(X_c.iloc[val_idx]))

            fold_diffs.append(acc_c - acc_b)
        differences.extend(fold_diffs)

    t_stat, p_value = stats.ttest_1samp(differences, 0.0)
    return {
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "significant": float(p_value) < 0.05,
        "mean_improvement": round(float(np.mean(differences)), 4),
        "interpretation": (
            "Model C significantly better than B (p < 0.05)"
            if float(p_value) < 0.05
            else "No significant difference between B and C (p >= 0.05)"
        ),
    }


def per_class_safety_analysis(
    X_symbolic: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Per-disease breakdown of biopsy triage safety.
    Shows % of correctly classified cases flagged as SAFE_BIOPSY_FREE.
    """
    from src.triage.biopsy_triage import BiopsyTriage
    triage = BiopsyTriage()
    triage_labels = triage.batch_recommend(X_symbolic)

    results = []
    for i, disease in enumerate(DISEASES):
        mask_true = (np.array(y_true) == i)
        mask_correct = (y_pred == i) & mask_true
        n_correct = int(mask_correct.sum())
        n_safe = int(((triage_labels == "SAFE_BIOPSY_FREE") & mask_correct).sum())
        pct_safe = (n_safe / n_correct * 100) if n_correct > 0 else 0.0
        results.append({
            "disease": disease,
            "n_patients": int(mask_true.sum()),
            "n_correct": n_correct,
            "n_safe_biopsy_free": n_safe,
            "pct_safe_biopsy_free": round(pct_safe, 1),
        })

    return pd.DataFrame(results)
