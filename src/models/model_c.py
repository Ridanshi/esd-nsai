import pandas as pd
from src.grading.fuzzy_grader import FuzzyGrader
from src.symbolic.pipeline import SymbolicPipeline
from src.models.base import cross_validate_model


def run_model_c(X_clinical: pd.DataFrame, y: pd.Series, rules_dir: str = "rules") -> dict:
    """
    Model C: 12 fuzzy clinical features + 9 symbolic outputs = 21 features.
    Novel hybrid contribution — combines clinical knowledge encoding with statistical learning.
    """
    grader = FuzzyGrader()
    X_fuzzy = grader.grade(X_clinical).reset_index(drop=True)

    pipeline = SymbolicPipeline(rules_dir)
    X_symbolic = pipeline.transform(X_fuzzy).reset_index(drop=True)

    X_combined = pd.concat([X_fuzzy, X_symbolic], axis=1)

    results = cross_validate_model(X_combined, y, label="Model C (12 clinical + 9 symbolic)")
    results["X_combined"] = X_combined
    return results
