import numpy as np
import pandas as pd

ENGINEERED_FEATURE_NAMES = [
    "pso_triad",
    "lp_classic",
    "prp_core",
    "itch_no_border",
    "young_adult",
    "older_patient",
    "scale_erythema_ratio",
    "inflammation_burden",
    "no_specific_morphology",
]


class FeatureEngineer:
    """
    Computes clinically-grounded interaction and derived features from fuzzy-graded
    clinical observations. Input must already be fuzzy-graded (0.0-1.0 range).

    All features are interpretable: each encodes a named clinical pattern.
    """

    def engineer_series(self, patient: pd.Series) -> pd.Series:
        koebner = float(patient.get("koebner_phenomenon", 0.0))
        knee = float(patient.get("knee_elbow_involvement", 0.0))
        family = float(patient.get("family_history", 0.0))
        polygonal = float(patient.get("polygonal_papules", 0.0))
        oral = float(patient.get("oral_mucosal_involvement", 0.0))
        follicular = float(patient.get("follicular_papules", 0.0))
        scaling = float(patient.get("scaling", 0.0))
        erythema = float(patient.get("erythema", 0.0))
        itching = float(patient.get("itching", 0.0))
        borders = float(patient.get("definite_borders", 0.0))
        scalp = float(patient.get("scalp_involvement", 0.0))
        age = float(patient.get("age", 0.0))

        return pd.Series({
            # Psoriasis: koebner + extensor distribution + genetic history
            "pso_triad": koebner * knee * family,
            # Lichen planus: polygonal papules + oral lesions (Wickham's striae)
            "lp_classic": polygonal * oral,
            # PRP: follicular papules + scaling
            "prp_core": follicular * scaling,
            # Chronic dermatitis: intense itch with ill-defined borders
            "itch_no_border": itching * (1.0 - borders),
            # Pityriasis rosea peaks 15-35 years (age < 0.4375 = 35/80)
            "young_adult": 1.0 if age < 0.4375 else 0.0,
            # Older patients: PRP adult type, late-onset psoriasis
            "older_patient": 1.0 if age > 0.625 else 0.0,
            # Psoriasis: scaling-dominant; LP/chronic derm: erythema-dominant
            "scale_erythema_ratio": scaling / (erythema + 0.01),
            # Overall inflammatory burden across the three cardinal symptoms
            "inflammation_burden": (erythema + scaling + itching) / 9.0,
            # No disease-specific morphology → seb derm or chronic derm territory
            "no_specific_morphology": 1.0 - float(np.clip(polygonal + follicular + koebner, 0.0, 1.0)),
        })

    def engineer(self, X_fuzzy: pd.DataFrame) -> pd.DataFrame:
        return X_fuzzy.apply(self.engineer_series, axis=1).reset_index(drop=True)
