import pytest
from src.symbolic.conflict import ConflictAnalyzer


@pytest.fixture
def analyzer():
    return ConflictAnalyzer()


def test_no_conflict_single_dominant(analyzer):
    scores = {
        "psoriasis": 0.9, "seborrheic_dermatitis": 0.05,
        "lichen_planus": 0.02, "pityriasis_rosea": 0.01,
        "chronic_dermatitis": 0.01, "pityriasis_rubra_pilaris": 0.01
    }
    load, severity = analyzer.analyze(scores)
    assert load < 0.1


def test_high_conflict_two_strong(analyzer):
    scores = {
        "psoriasis": 0.8, "seborrheic_dermatitis": 0.75,
        "lichen_planus": 0.0, "pityriasis_rosea": 0.0,
        "chronic_dermatitis": 0.0, "pityriasis_rubra_pilaris": 0.0
    }
    load, _ = analyzer.analyze(scores)
    assert load > 0.3


def test_contradiction_severity_incompatible_pair(analyzer):
    scores = {
        "psoriasis": 0.8, "lichen_planus": 0.7,
        "seborrheic_dermatitis": 0.0, "pityriasis_rosea": 0.0,
        "chronic_dermatitis": 0.0, "pityriasis_rubra_pilaris": 0.0
    }
    _, severity = analyzer.analyze(scores)
    assert severity > 0.4


def test_outputs_in_range(analyzer):
    scores = {d: 0.5 for d in [
        "psoriasis", "seborrheic_dermatitis", "lichen_planus",
        "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris"
    ]}
    load, severity = analyzer.analyze(scores)
    assert 0.0 <= load <= 1.0
    assert 0.0 <= severity <= 1.0


def test_all_zero_no_conflict(analyzer):
    scores = {d: 0.0 for d in [
        "psoriasis", "seborrheic_dermatitis", "lichen_planus",
        "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris"
    ]}
    load, severity = analyzer.analyze(scores)
    assert load == pytest.approx(0.0)
    assert severity == pytest.approx(0.0)
