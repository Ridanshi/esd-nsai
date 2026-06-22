import pandas as pd
from src.models.base import cross_validate_model


def run_model_a(X_all: pd.DataFrame, y: pd.Series) -> dict:
    """
    Model A: all 34 features (clinical + histopathological).
    Biopsy-assisted upper bound — replicates existing literature.
    """
    return cross_validate_model(X_all, y, label="Model A (All 34 features)")
