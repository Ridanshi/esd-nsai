import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

N_SPLITS = 10
RANDOM_STATE = 42

DISEASES = [
    "psoriasis", "seborrheic_dermatitis", "lichen_planus",
    "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris"
]


def get_xgb_params() -> dict:
    return {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
    }


def get_xgb_params_c() -> dict:
    """Regularised params for Model C (21 features, 366 samples — higher overfit risk)."""
    return {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_weight": 3,
        "reg_lambda": 3.0,
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
    }


def cross_validate_model(X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    """
    Stratified 10-fold CV with XGBoost. Returns aggregated metrics.
    y must be 0-indexed (0-5).
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accuracies, macro_f1s = [], []
    all_y_true, all_y_pred = [], []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**get_xgb_params())
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
