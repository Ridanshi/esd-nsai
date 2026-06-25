import numpy as np
import pandas as pd

# young_adult dropped: MI=0.0005 (near-zero contribution, age signal already in symbolic layer)
ENGINEERED_FEATURE_NAMES = [
    "pso_triad",
    "lp_classic",
    "prp_core",
    "itch_no_border",
    "older_patient",
    "scale_erythema_ratio",
    "inflammation_burden",
    "no_specific_morphology",
]


class FeatureEngineer:
    """
    Computes 8 clinically-grounded interaction and derived features from
    fuzzy-graded clinical observations. Input must already be fuzzy-graded (0-1).

    Features selected via mutual information scoring (threshold MI >= 0.05).
    All features encode named clinical patterns and are fully interpretable.
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
        age = float(patient.get("age", 0.0))

        return pd.Series({
            # Psoriasis: koebner + extensor distribution + genetic history
            "pso_triad": koebner * knee * family,
            # Lichen planus: polygonal papules + oral Wickham's striae (MI=0.394)
            "lp_classic": polygonal * oral,
            # PRP: follicular papules + scaling
            "prp_core": follicular * scaling,
            # Chronic dermatitis: intense itch with ill-defined borders
            "itch_no_border": itching * (1.0 - borders),
            # PRP adult type, late-onset psoriasis — age > 50yr (MI=0.081)
            "older_patient": 1.0 if age > 0.625 else 0.0,
            # Psoriasis = scaling-dominant; LP/chronic derm = erythema-dominant (MI=0.144)
            "scale_erythema_ratio": scaling / (erythema + 0.01),
            # Overall inflammatory burden: (erythema + scaling + itching) / max (MI=0.164)
            "inflammation_burden": (erythema + scaling + itching) / 9.0,
            # No disease-specific morphology → seb derm or chronic derm territory (MI=0.345)
            "no_specific_morphology": 1.0 - float(np.clip(polygonal + follicular + koebner, 0.0, 1.0)),
        })

    def engineer(self, X_fuzzy: pd.DataFrame) -> pd.DataFrame:
        return X_fuzzy.apply(self.engineer_series, axis=1).reset_index(drop=True)
