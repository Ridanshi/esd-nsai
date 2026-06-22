import pytest
import pandas as pd
from src.symbolic.rule_engine import RuleEngine

RULES_DIR = "rules"


@pytest.fixture
def engine():
    return RuleEngine(RULES_DIR)


@pytest.fixture
def psoriasis_patient():
    return pd.Series({
        "erythema": 0.6667,
        "scaling": 0.6667,
        "definite_borders": 0.3333,
        "itching": 0.3333,
        "koebner_phenomenon": 1.0,
        "polygonal_papules": 0.0,
        "follicular_papules": 0.0,
        "oral_mucosal_involvement": 0.0,
        "knee_elbow_involvement": 1.0,
        "scalp_involvement": 1.0,
        "family_history": 0.0,
        "age": 0.4375,
    })


@pytest.fixture
def minimal_patient():
    return pd.Series({feat: 0.0 for feat in [
        "erythema", "scaling", "definite_borders", "itching",
        "koebner_phenomenon", "polygonal_papules", "follicular_papules",
        "oral_mucosal_involvement", "knee_elbow_involvement",
        "scalp_involvement", "family_history", "age"
    ]})


def test_fire_returns_all_diseases(engine, psoriasis_patient):
    scores = engine.fire(psoriasis_patient)
    expected = {
        "psoriasis", "seborrheic_dermatitis", "lichen_planus",
        "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris"
    }
    assert set(scores.keys()) == expected


def test_fire_values_in_range(engine, psoriasis_patient):
    scores = engine.fire(psoriasis_patient)
    for disease, score in scores.items():
        assert 0.0 <= score <= 1.0, f"{disease} score {score} out of range"


def test_psoriasis_high_for_psoriasis_patient(engine, psoriasis_patient):
    scores = engine.fire(psoriasis_patient)
    assert scores["psoriasis"] > 0.5
    assert scores["psoriasis"] == max(scores.values())


def test_minimal_patient_all_zero(engine, minimal_patient):
    scores = engine.fire(minimal_patient)
    for score in scores.values():
        assert score == pytest.approx(0.0)


def test_fired_rules_structure(engine, psoriasis_patient):
    fired = engine.get_fired_rules(psoriasis_patient)
    assert len(fired) > 0
    for rule in fired:
        assert "id" in rule
        assert "disease" in rule
        assert "firing_strength" in rule
        assert "contribution" in rule
        assert 0.0 <= rule["firing_strength"] <= 1.0
