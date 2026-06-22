import pytest
import pandas as pd
import numpy as np
from src.grading.fuzzy_grader import FuzzyGrader

ORDINAL_FEATURES = ["erythema", "scaling", "definite_borders", "itching"]
BINARY_FEATURES = [
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_elbow_involvement",
    "scalp_involvement", "family_history"
]


def test_ordinal_grading():
    grader = FuzzyGrader()
    row = pd.Series({"erythema": 0, "scaling": 1, "definite_borders": 2, "itching": 3})
    result = grader.grade_series(row)
    assert result["erythema"] == pytest.approx(0.0)
    assert result["scaling"] == pytest.approx(1 / 3, abs=1e-4)
    assert result["definite_borders"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["itching"] == pytest.approx(1.0)


def test_binary_features_unchanged():
    grader = FuzzyGrader()
    row = pd.Series({"koebner_phenomenon": 1, "family_history": 0})
    result = grader.grade_series(row)
    assert result["koebner_phenomenon"] == 1.0
    assert result["family_history"] == 0.0


def test_age_normalized():
    grader = FuzzyGrader(age_max=80)
    row = pd.Series({"age": 40})
    result = grader.grade_series(row)
    assert result["age"] == pytest.approx(0.5)


def test_output_range():
    grader = FuzzyGrader()
    data = pd.DataFrame([{
        "erythema": 2, "scaling": 1, "definite_borders": 0, "itching": 3,
        "koebner_phenomenon": 1, "polygonal_papules": 0, "follicular_papules": 0,
        "oral_mucosal_involvement": 0, "knee_elbow_involvement": 1,
        "scalp_involvement": 1, "family_history": 0, "age": 35,
    }])
    result = grader.grade(data)
    assert result.min().min() >= 0.0
    assert result.max().max() <= 1.0


def test_dataframe_shape_preserved():
    grader = FuzzyGrader()
    data = pd.DataFrame([
        {"erythema": 1, "scaling": 2, "age": 30},
        {"erythema": 3, "scaling": 0, "age": 60},
    ])
    result = grader.grade(data)
    assert result.shape == data.shape
