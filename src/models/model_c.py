import pandas as pd
from xgboost import XGBClassifier
from src.grading.fuzzy_grader import FuzzyGrader
from src.symbolic.pipeline import SymbolicPipeline
from src.models.base import cross_validate_model, get_xgb_params_c, RANDOM_STATE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from src.models.base import DISEASES, N_SPLITS


def run_model_c(X_clinical: pd.DataFrame, y: pd.Series, rules_dir: str = "rules") -> dict:
    """
    Model C: 12 fuzzy clinical features + 9 symbolic outputs = 21 features.
    Uses regularised XGBoost params to reduce overfit on the expanded feature set.
    """
    grader = FuzzyGrader()
    X_fuzzy = grader.grade(X_clinical).reset_index(drop=True)

    pipeline = SymbolicPipeline(rules_dir)
    X_symbolic = pipeline.transform(X_fuzzy).reset_index(drop=True)

    X_combined = pd.concat([X_fuzzy, X_symbolic], axis=1)

    results = _cross_validate_c(X_combined, y, label="Model C (12 clinical + 9 symbolic)")
    results["X_combined"] = X_combined
    return results


def _cross_validate_c(X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    """Stratified 10-fold CV using regularised params specific to Model C."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accuracies, macro_f1s = [], []
    all_y_true, all_y_pred = [], []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**get_xgb_params_c())
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_val)

        accuracies.append(accuracy_score(y_val, y_pred))
        macro_f1s.append(f1_score(y_val, y_pred, average="macro", zero_division=0))
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())

    per_class_f1 = f1_score(all_y_true, all_y_pred, average=None, zero_division=0)
    cm = confusion_matrix(all_y_true, all_y_pred)

    return {
        "label": label,
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "per_class_f1": {DISEASES[i]: round(float(per_class_f1[i]), 4)
                         for i in range(len(DISEASES))},
        "confusion_matrix": cm,
    }
