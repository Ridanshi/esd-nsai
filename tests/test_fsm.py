import pytest
from src.symbolic.fsm import DiagnosticFSM, FSMState


@pytest.fixture
def fsm():
    return DiagnosticFSM()


def test_all_zero_evidence_sparse(fsm):
    scores = {d: 0.0 for d in [
        "psoriasis", "seborrheic_dermatitis", "lichen_planus",
        "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris"
    ]}
    state = fsm.traverse(scores, conflict_load=0.0)
    assert state == FSMState.EVIDENCE_SPARSE


def test_low_certainty_hypothesis_forming(fsm):
    scores = {
        "psoriasis": 0.2, "seborrheic_dermatitis": 0.0, "lichen_planus": 0.0,
        "pityriasis_rosea": 0.0, "chronic_dermatitis": 0.0, "pityriasis_rubra_pilaris": 0.0
    }
    state = fsm.traverse(scores, conflict_load=0.0)
    assert state == FSMState.HYPOTHESIS_FORMING


def test_moderate_certainty_building(fsm):
    scores = {
        "psoriasis": 0.5, "seborrheic_dermatitis": 0.0, "lichen_planus": 0.0,
        "pityriasis_rosea": 0.0, "chronic_dermatitis": 0.0, "pityriasis_rubra_pilaris": 0.0
    }
    state = fsm.traverse(scores, conflict_load=0.1)
    assert state == FSMState.CERTAINTY_BUILDING


def test_high_conflict_diagnostic_tension(fsm):
    scores = {
        "psoriasis": 0.6, "seborrheic_dermatitis": 0.55, "lichen_planus": 0.0,
        "pityriasis_rosea": 0.0, "chronic_dermatitis": 0.0, "pityriasis_rubra_pilaris": 0.0
    }
    state = fsm.traverse(scores, conflict_load=0.35)
    assert state == FSMState.DIAGNOSTIC_TENSION


def test_high_certainty_low_conflict_resolved(fsm):
    scores = {
        "psoriasis": 0.85, "seborrheic_dermatitis": 0.05, "lichen_planus": 0.0,
        "pityriasis_rosea": 0.0, "chronic_dermatitis": 0.0, "pityriasis_rubra_pilaris": 0.0
    }
    state = fsm.traverse(scores, conflict_load=0.05)
    assert state == FSMState.RESOLVED


def test_state_is_int(fsm):
    scores = {
        "psoriasis": 0.9, "seborrheic_dermatitis": 0.0, "lichen_planus": 0.0,
        "pityriasis_rosea": 0.0, "chronic_dermatitis": 0.0, "pityriasis_rubra_pilaris": 0.0
    }
    state = fsm.traverse(scores, 0.0)
    assert isinstance(state, int)
    assert 0 <= state <= 4
