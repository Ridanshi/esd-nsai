import pytest
import pandas as pd
import numpy as np
from src.grading.feature_engineer import FeatureEngineer, ENGINEERED_FEATURE_NAMES


@pytest.fixture
def engineer():
    return FeatureEngineer()


@pytest.fixture
def psoriasis_fuzzy():
    return pd.Series({
        "erythema": 0.6667, "scaling": 0.6667, "definite_borders": 0.3333,
        "itching": 0.3333, "koebner_phenomenon": 1.0, "polygonal_papules": 0.0,
        "follicular_papules": 0.0, "oral_mucosal_involvement": 0.0,
        "knee_elbow_involvement": 1.0, "scalp_involvement": 1.0,
        "family_history": 1.0, "age": 0.375,
    })


@pytest.fixture
def lp_fuzzy():
    return pd.Series({
        "erythema": 0.6667, "scaling": 0.3333, "definite_borders": 0.3333,
        "itching": 1.0, "koebner_phenomenon": 1.0, "polygonal_papules": 1.0,
        "follicular_papules": 0.0, "oral_mucosal_involvement": 1.0,
        "knee_elbow_involvement": 0.0, "scalp_involvement": 0.0,
        "family_history": 0.0, "age": 0.5,
    })


@pytest.fixture
def zero_patient():
    return pd.Series({f: 0.0 for f in [
        "erythema", "scaling", "definite_borders", "itching",
        "koebner_phenomenon", "polygonal_papules", "follicular_papules",
        "oral_mucosal_involvement", "knee_elbow_involvement",
        "scalp_involvement", "family_history", "age",
    ]})


def test_engineer_series_returns_all_features(engineer, psoriasis_fuzzy):
    result = engineer.engineer_series(psoriasis_fuzzy)
    assert set(result.index) == set(ENGINEERED_FEATURE_NAMES)


def test_pso_triad_fires_for_psoriasis(engineer, psoriasis_fuzzy):
    result = engineer.engineer_series(psoriasis_fuzzy)
    assert result["pso_triad"] == pytest.approx(1.0)


def test_lp_classic_fires_for_lp(engineer, lp_fuzzy):
    result = engineer.engineer_series(lp_fuzzy)
    assert result["lp_classic"] == pytest.approx(1.0)


def test_pso_triad_zero_when_no_family_history(engineer, psoriasis_fuzzy):
    patient = psoriasis_fuzzy.copy()
    patient["family_history"] = 0.0
    result = engineer.engineer_series(patient)
    assert result["pso_triad"] == pytest.approx(0.0)


def test_young_adult_flag(engineer, zero_patient):
    young = zero_patient.copy()
    young["age"] = 0.30   # 24 years
    old = zero_patient.copy()
    old["age"] = 0.70     # 56 years
    assert engineer.engineer_series(young)["young_adult"] == 1.0
    assert engineer.engineer_series(old)["young_adult"] == 0.0


def test_no_specific_morphology_all_zero(engineer, zero_patient):
    result = engineer.engineer_series(zero_patient)
    assert result["no_specific_morphology"] == pytest.approx(1.0)


def test_engineer_dataframe_shape(engineer, psoriasis_fuzzy):
    X = pd.DataFrame([psoriasis_fuzzy] * 5)
    result = engineer.engineer(X)
    assert result.shape == (5, len(ENGINEERED_FEATURE_NAMES))


def test_all_values_finite(engineer, psoriasis_fuzzy):
    result = engineer.engineer_series(psoriasis_fuzzy)
    assert all(np.isfinite(v) for v in result.values)
