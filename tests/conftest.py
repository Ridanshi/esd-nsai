import pytest
import pandas as pd
import numpy as np

CLINICAL_FEATURES = [
    "erythema", "scaling", "definite_borders", "itching",
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_elbow_involvement",
    "scalp_involvement", "family_history", "age"
]

BINARY_FEATURES = [
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_elbow_involvement",
    "scalp_involvement", "family_history"
]

ORDINAL_FEATURES = [
    "erythema", "scaling", "definite_borders", "itching"
]

DISEASES = [
    "psoriasis", "seborrheic_dermatitis", "lichen_planus",
    "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris"
]

CLASS_MAP = {
    1: "psoriasis",
    2: "seborrheic_dermatitis",
    3: "lichen_planus",
    4: "pityriasis_rosea",
    5: "chronic_dermatitis",
    6: "pityriasis_rubra_pilaris"
}

@pytest.fixture
def sample_patient_ordinal():
    """A psoriasis-like patient with raw ordinal values."""
    return pd.Series({
        "erythema": 2,
        "scaling": 2,
        "definite_borders": 1,
        "itching": 1,
        "koebner_phenomenon": 1,
        "polygonal_papules": 0,
        "follicular_papules": 0,
        "oral_mucosal_involvement": 0,
        "knee_elbow_involvement": 1,
        "scalp_involvement": 1,
        "family_history": 0,
        "age": 35,
    })

@pytest.fixture
def sample_patient_fuzzy():
    """Same patient after fuzzy grading."""
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
        "age": 0.4375,  # 35 / 80
    })
