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


def test_contradiction_blocks_safe(triage):
    """High contradiction_severity must veto SAFE_BIOPSY_FREE even when all else qualifies."""
    kwargs = dict(top_certainty=0.80, conflict_load=0.10, fsm_state=FSMState.RESOLVED)
    assert triage.recommend(**kwargs, contradiction_severity=0.10) == "SAFE_BIOPSY_FREE"
    assert triage.recommend(**kwargs, contradiction_severity=0.50) != "SAFE_BIOPSY_FREE"


def test_contradiction_defaults_to_permissive(triage):
    """Omitting contradiction_severity must not silently block SAFE_BIOPSY_FREE."""
    result = triage.recommend(
        top_certainty=0.80, conflict_load=0.10, fsm_state=FSMState.RESOLVED
    )
    assert result == "SAFE_BIOPSY_FREE"


def test_batch_recommend_reads_contradiction_column(triage):
    """batch_recommend must actually consult contradiction_severity, not ignore it."""
    base = {
        "certainty_psoriasis": 0.85, "certainty_seborrheic_dermatitis": 0.0,
        "certainty_lichen_planus": 0.0, "certainty_pityriasis_rosea": 0.0,
        "certainty_chronic_dermatitis": 0.0, "certainty_pityriasis_rubra_pilaris": 0.0,
        "conflict_load": 0.05, "fsm_state": 4,
    }
    data = pd.DataFrame([
        {**base, "contradiction_severity": 0.0},
        {**base, "contradiction_severity": 0.60},
    ])
    result = triage.batch_recommend(data)
    assert result.iloc[0] == "SAFE_BIOPSY_FREE"
    assert result.iloc[1] != "SAFE_BIOPSY_FREE"


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
