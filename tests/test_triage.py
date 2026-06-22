import pytest
import pandas as pd
from src.triage.biopsy_triage import BiopsyTriage, TRIAGE_TIERS
from src.symbolic.fsm import FSMState


@pytest.fixture
def triage():
    return BiopsyTriage()


def test_safe_biopsy_free(triage):
    result = triage.recommend(
        top_certainty=0.80,
        conflict_load=0.10,
        fsm_state=FSMState.RESOLVED
    )
    assert result == "SAFE_BIOPSY_FREE"


def test_uncertain(triage):
    result = triage.recommend(
        top_certainty=0.60,
        conflict_load=0.25,
        fsm_state=FSMState.CERTAINTY_BUILDING
    )
    assert result == "UNCERTAIN"


def test_biopsy_advised_low_certainty(triage):
    result = triage.recommend(
        top_certainty=0.30,
        conflict_load=0.50,
        fsm_state=FSMState.DIAGNOSTIC_TENSION
    )
    assert result == "BIOPSY_ADVISED"


def test_biopsy_advised_high_conflict(triage):
    result = triage.recommend(
        top_certainty=0.80,
        conflict_load=0.45,
        fsm_state=FSMState.RESOLVED
    )
    assert result != "SAFE_BIOPSY_FREE"


def test_batch_recommend_length(triage):
    data = pd.DataFrame([
        {
            "certainty_psoriasis": 0.85, "certainty_seborrheic_dermatitis": 0.0,
            "certainty_lichen_planus": 0.0, "certainty_pityriasis_rosea": 0.0,
            "certainty_chronic_dermatitis": 0.0, "certainty_pityriasis_rubra_pilaris": 0.0,
            "conflict_load": 0.05, "contradiction_severity": 0.0, "fsm_state": 4,
        },
        {
            "certainty_psoriasis": 0.40, "certainty_seborrheic_dermatitis": 0.0,
            "certainty_lichen_planus": 0.0, "certainty_pityriasis_rosea": 0.0,
            "certainty_chronic_dermatitis": 0.0, "certainty_pityriasis_rubra_pilaris": 0.0,
            "conflict_load": 0.60, "contradiction_severity": 0.0, "fsm_state": 3,
        },
    ])
    result = triage.batch_recommend(data)
    assert len(result) == 2
    assert set(result.unique()).issubset(set(TRIAGE_TIERS))
