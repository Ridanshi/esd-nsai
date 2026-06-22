import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

CLINICAL_FEATURES = [
    "erythema", "scaling", "definite_borders", "itching",
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_elbow_involvement",
    "scalp_involvement", "family_history", "age"
]

HISTOPATH_FEATURES = [
    "melanin_incontinence", "eosinophils_in_the_infiltrate",
    "pnl_infiltrate", "fibrosis_of_the_papillary_dermis",
    "exocytosis", "acanthosis", "hyperkeratosis", "parakeratosis",
    "clubbing_of_the_rete_ridges", "elongation_of_the_rete_ridges",
    "thinning_of_the_suprapapillary_epidermis", "spongiform_pustule",
    "munro_microabcess", "focal_hypergranulosis",
    "disappearance_of_the_granular_layer",
    "vacuolisation_and_damage_of_the_basal_layer", "spongiosis",
    "saw_tooth_appearance_of_retes", "follicular_horn_plug",
    "perifollicular_parakeratosis", "inflammatory_monoluclear_infiltrate",
    "band_like_infiltrate"
]

CLASS_MAP = {
    1: "psoriasis",
    2: "seborrheic_dermatitis",
    3: "lichen_planus",
    4: "pityriasis_rosea",
    5: "chronic_dermatitis",
    6: "pityriasis_rubra_pilaris"
}

_CACHE = {}


def load_dataset() -> tuple:
    """
    Returns (X_clinical, X_histopath, X_all, y).
    y values are 0-indexed (0-5). Results cached after first fetch.
    Missing age values filled with median.
    """
    if _CACHE:
        return _CACHE["result"]

    raw = fetch_ucirepo(id=33)
    X = raw.data.features.copy()
    y_raw = raw.data.targets.iloc[:, 0]

    # Standardise column names: lowercase, spaces/hyphens -> underscores
    X.columns = [c.lower().replace(" ", "_").replace("-", "_") for c in X.columns]

    # Median imputation for missing age values
    if "age" in X.columns and X["age"].isnull().any():
        X["age"] = X["age"].fillna(X["age"].median())

    # Select feature subsets — use canonical order from CLINICAL_FEATURES
    clinical_cols = [c for c in CLINICAL_FEATURES if c in X.columns]
    histopath_cols = [c for c in X.columns if c not in clinical_cols]

    X_clinical = X[clinical_cols].reset_index(drop=True)
    X_histopath = X[histopath_cols].reset_index(drop=True)
    X_all = X.reset_index(drop=True)

    # Encode targets 1-6 → 0-5
    y = pd.Series(
        y_raw.values.ravel().astype(int) - 1,
        name="disease"
    )

    result = (X_clinical, X_histopath, X_all, y)
    _CACHE["result"] = result
    return result
