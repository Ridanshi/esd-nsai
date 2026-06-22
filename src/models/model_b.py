import pandas as pd
from src.models.base import cross_validate_model
from src.grading.fuzzy_grader import FuzzyGrader


def run_model_b(X_clinical: pd.DataFrame, y: pd.Series) -> dict:
    """
    Model B: 12 clinical features only (biopsy-free baseline).
    Features are fuzzy-graded before classification.
    """
    grader = FuzzyGrader()
    X_fuzzy = grader.grade(X_clinical)
    return cross_validate_model(X_fuzzy, y, label="Model B (12 clinical features)")
