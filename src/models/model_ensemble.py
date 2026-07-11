import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.grading.fuzzy_grader import FuzzyGrader
from src.grading.feature_engineer import FeatureEngineer
from src.symbolic.pipeline import SymbolicPipeline
from src.models.base import get_xgb_params_c, RANDOM_STATE, DISEASES, N_SPLITS


def _build_ensemble() -> VotingClassifier:
    xgb = XGBClassifier(**get_xgb_params_c())
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1,
    )
    lgbm = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=15,
        min_child_samples=10, reg_lambda=5.0,
        random_state=RANDOM_STATE, verbose=-1,
    )
    return VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf), ("lgbm", lgbm)],
        voting="soft",
    )


def run_model_ensemble(X_clinical: pd.DataFrame, y: pd.Series, rules_dir: str = "rules") -> dict:
    """
    Ensemble Model D: soft-vote XGBoost + Random Forest + LightGBM on 29 features.
    Biopsy-free. Same feature pipeline as Model C.
    """
    grader = FuzzyGrader()
    X_fuzzy = grader.grade(X_clinical).reset_index(drop=True)
    engineer = FeatureEngineer()
    X_engineered = engineer.engineer(X_fuzzy)
    pipeline = SymbolicPipeline(rules_dir)
    X_symbolic = pipeline.transform(X_fuzzy).reset_index(drop=True)
    X = pd.concat([X_fuzzy, X_engineered, X_symbolic], axis=1)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accuracies, macro_f1s = [], []
    all_y_true, all_y_pred = [], []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = _build_ensemble()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        accuracies.append(accuracy_score(y_val, y_pred))
        macro_f1s.append(f1_score(y_val, y_pred, average="macro", zero_division=0))
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())

    per_class_f1 = f1_score(all_y_true, all_y_pred, average=None, zero_division=0)
    cm = confusion_matrix(all_y_true, all_y_pred)

    return {
        "label": "Model D (Ensemble: XGB + RF + LGBM, 29 features)",
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
