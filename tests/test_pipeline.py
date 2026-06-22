import pytest
import pandas as pd
from src.symbolic.pipeline import SymbolicPipeline, SYMBOLIC_FEATURE_NAMES

RULES_DIR = "rules"


@pytest.fixture
def pipeline():
    return SymbolicPipeline(RULES_DIR)


@pytest.fixture
def two_patients():
    return pd.DataFrame([
        {
            "erythema": 0.6667, "scaling": 0.6667, "definite_borders": 0.3333,
            "itching": 0.3333, "koebner_phenomenon": 1.0, "polygonal_papules": 0.0,
            "follicular_papules": 0.0, "oral_mucosal_involvement": 0.0,
            "knee_elbow_involvement": 1.0, "scalp_involvement": 1.0,
            "family_history": 0.0, "age": 0.4375,
        },
        {
            "erythema": 0.0, "scaling": 0.0, "definite_borders": 0.0,
            "itching": 0.0, "koebner_phenomenon": 0.0, "polygonal_papules": 0.0,
            "follicular_papules": 0.0, "oral_mucosal_involvement": 0.0,
            "knee_elbow_involvement": 0.0, "scalp_involvement": 0.0,
            "family_history": 0.0, "age": 0.0,
        },
    ])


def test_transform_shape(pipeline, two_patients):
    result = pipeline.transform(two_patients)
    assert result.shape == (2, 9)


def test_transform_column_names(pipeline, two_patients):
    result = pipeline.transform(two_patients)
    assert list(result.columns) == SYMBOLIC_FEATURE_NAMES


def test_transform_value_range(pipeline, two_patients):
    result = pipeline.transform(two_patients)
    assert result.min().min() >= 0.0
    assert result.drop(columns=["fsm_state"]).max().max() <= 1.0
    assert result["fsm_state"].max() <= 4


def test_explain_keys(pipeline, two_patients):
    trace = pipeline.explain(two_patients.iloc[0])
    assert "certainty_scores" in trace
    assert "conflict_load" in trace
    assert "contradiction_severity" in trace
    assert "fsm_state" in trace
    assert "fired_rules" in trace
