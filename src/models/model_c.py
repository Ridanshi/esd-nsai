import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from src.grading.fuzzy_grader import FuzzyGrader
from src.grading.feature_engineer import FeatureEngineer
from src.symbolic.pipeline import SymbolicPipeline
from src.models.base import get_catboost_params_c, RANDOM_STATE, DISEASES, N_SPLITS
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def run_model_c(X_clinical: pd.DataFrame, y: pd.Series, rules_dir: str = "rules") -> dict:
    """
    Model C: 12 fuzzy + 8 engineered + 9 symbolic = 29 features.
    Pipeline: raw -> FuzzyGrader -> FeatureEngineer -> SymbolicPipeline -> CatBoost.
    """
    grader = FuzzyGrader()
    X_fuzzy = grader.grade(X_clinical).reset_index(drop=True)

    engineer = FeatureEngineer()
    X_engineered = engineer.engineer(X_fuzzy)

    pipeline = SymbolicPipeline(rules_dir)
    X_symbolic = pipeline.transform(X_fuzzy).reset_index(drop=True)

    X_combined = pd.concat([X_fuzzy, X_engineered, X_symbolic], axis=1)

    results = _cross_validate_c(X_combined, y, label="Model C — HSCIS-ESD (29 features, CatBoost)")
    results["X_combined"] = X_combined
    return results


def _cross_validate_c(X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    """Stratified 10-fold CV with CatBoost for Model C."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accuracies, macro_f1s = [], []
    all_y_true, all_y_pred = [], []

    for train_idx, val_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        model = CatBoostClassifier(**get_catboost_params_c())
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val).flatten()

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
        "y_true_cv": all_y_true,
        "y_pred_cv": all_y_pred,
    }
