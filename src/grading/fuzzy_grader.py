import pandas as pd
import numpy as np

ORDINAL_FEATURES = ["erythema", "scaling", "definite_borders", "itching"]
BINARY_FEATURES = [
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_elbow_involvement",
    "scalp_involvement", "family_history"
]
ORDINAL_MAX = 3.0


class FuzzyGrader:
    def __init__(self, age_max: float = 80.0):
        self.age_max = age_max

    def grade_series(self, row: pd.Series) -> pd.Series:
        result = row.copy().astype(float)
        for feat in ORDINAL_FEATURES:
            if feat in result.index:
                result[feat] = float(result[feat]) / ORDINAL_MAX
        if "age" in result.index:
            result["age"] = min(float(result["age"]) / self.age_max, 1.0)
        # Binary features already 0 or 1 — no transform needed
        return result.clip(0.0, 1.0)

    def grade(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.apply(self.grade_series, axis=1)
