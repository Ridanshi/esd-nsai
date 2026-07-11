import numpy as np
import pandas as pd
from scipy.stats import chi2
from src.models.base import DISEASES


def print_comparison_table(results_a: dict, results_b: dict, results_c: dict) -> None:
    header = f"{'Model':<46} {'Accuracy':>16} {'Macro F1':>16}"
    print("\n" + "=" * 82)
    print(header)
    print("-" * 82)
    for res in [results_a, results_b, results_c]:
        label = res["label"]
        acc = f"{res['accuracy_mean']:.4f} +/- {res['accuracy_std']:.4f}"
        f1 = f"{res['macro_f1_mean']:.4f} +/- {res['macro_f1_std']:.4f}"
        print(f"{label:<46} {acc:>16} {f1:>16}")
    print("=" * 82)

    print("\nPer-class F1 scores:")
    print(f"{'Disease':<35} {'Model A':>10} {'Model B':>10} {'Model C':>10}")
    print("-" * 67)
    for disease in DISEASES:
        fa = results_a["per_class_f1"].get(disease, 0.0)
        fb = results_b["per_class_f1"].get(disease, 0.0)
        fc = results_c["per_class_f1"].get(disease, 0.0)
        print(f"{disease:<35} {fa:>10.4f} {fb:>10.4f} {fc:>10.4f}")


def run_mcnemar_test(res_b: dict, res_c: dict) -> dict:
    """
    McNemar's test comparing Model B vs Model C on per-sample 10-fold CV predictions.
    More appropriate than 5x2 t-test for small datasets (N=366): operates at the
    prediction level rather than the fold level, so training-set size does not bias it.

    Contingency table:
        a = both correct   b = B correct, C wrong
        c = B wrong, C correct   d = both wrong
    Test statistic: chi2 = (|b - c| - 1)^2 / (b + c)  [with Yates continuity correction]
    """
    y_true = np.array(res_b["y_true_cv"])
    y_pred_b = np.array(res_b["y_pred_cv"])
    y_pred_c = np.array(res_c["y_pred_cv"])

    b_correct = y_pred_b == y_true
    c_correct = y_pred_c == y_true

    a = int(np.sum(b_correct & c_correct))
    b = int(np.sum(b_correct & ~c_correct))
    c = int(np.sum(~b_correct & c_correct))
    d = int(np.sum(~b_correct & ~c_correct))

    if b + c == 0:
        return {
            "a": a, "b": b, "c": c, "d": d,
            "chi2_stat": 0.0, "p_value": 1.0,
            "significant": False,
            "c_wins": 0, "b_wins": 0,
            "interpretation": "Models produce identical predictions — no test possible.",
        }

    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1 - chi2.cdf(chi2_stat, df=1))

    return {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "chi2_stat": round(chi2_stat, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "c_wins": c,
        "b_wins": b,
        "interpretation": (
            f"Model C significantly better than B (p={p_value:.4f} < 0.05): "
            f"C correct on {c} cases B missed; B correct on {b} cases C missed."
            if p_value < 0.05
            else f"No significant difference (p={p_value:.4f}): "
                 f"C correct on {c} cases B missed; B correct on {b} cases C missed."
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
