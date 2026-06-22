import pytest
import pandas as pd
from src.data.loader import load_dataset, CLINICAL_FEATURES, HISTOPATH_FEATURES, CLASS_MAP


def test_dataset_shapes():
    X_clinical, X_histopath, X_all, y = load_dataset()
    assert X_clinical.shape == (366, 12)
    assert X_histopath.shape[0] == 366
    assert X_all.shape == (366, 34)
    assert len(y) == 366


def test_clinical_feature_names():
    X_clinical, _, _, _ = load_dataset()
    assert list(X_clinical.columns) == CLINICAL_FEATURES


def test_target_classes():
    _, _, _, y = load_dataset()
    assert set(y.unique()) == {0, 1, 2, 3, 4, 5}


def test_no_missing_values_after_imputation():
    X_clinical, _, _, _ = load_dataset()
    assert X_clinical.isnull().sum().sum() == 0


def test_age_range():
    X_clinical, _, _, _ = load_dataset()
    assert X_clinical["age"].min() >= 0
    assert X_clinical["age"].max() <= 120
