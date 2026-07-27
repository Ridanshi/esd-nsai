import pandas as pd
from src.symbolic.fsm import FSMState

TRIAGE_TIERS = ["SAFE_BIOPSY_FREE", "UNCERTAIN", "BIOPSY_ADVISED"]

SAFE_CERTAINTY_THRESHOLD = 0.75
SAFE_CONFLICT_THRESHOLD = 0.20
# Placeholder pending Ridanshi's clinical judgment (changes-made.md #7) — mirrors
# the existing SAFE_CONFLICT_THRESHOLD pattern, not derived from outcome data.
SAFE_CONTRADICTION_THRESHOLD = 0.30
UNCERTAIN_CERTAINTY_THRESHOLD = 0.55
UNCERTAIN_CONFLICT_THRESHOLD = 0.40


class BiopsyTriage:
    def recommend(
        self,
        top_certainty: float,
        conflict_load: float,
        fsm_state: int,
        contradiction_severity: float = 0.0,
    ) -> str:
        if (
            top_certainty >= SAFE_CERTAINTY_THRESHOLD
            and conflict_load < SAFE_CONFLICT_THRESHOLD
            and contradiction_severity < SAFE_CONTRADICTION_THRESHOLD
            and fsm_state == FSMState.RESOLVED
        ):
            return "SAFE_BIOPSY_FREE"
        if (
            top_certainty >= UNCERTAIN_CERTAINTY_THRESHOLD
            and conflict_load < UNCERTAIN_CONFLICT_THRESHOLD
        ):
            return "UNCERTAIN"
        return "BIOPSY_ADVISED"

    def batch_recommend(self, X_symbolic: pd.DataFrame) -> pd.Series:
        disease_certainty_cols = [
            c for c in X_symbolic.columns if c.startswith("certainty_")
        ]
        top_certainty = X_symbolic[disease_certainty_cols].max(axis=1)
        return pd.Series([
            self.recommend(
                top_certainty=top_certainty.iloc[i],
                conflict_load=float(X_symbolic["conflict_load"].iloc[i]),
                fsm_state=int(X_symbolic["fsm_state"].iloc[i]),
                contradiction_severity=float(X_symbolic["contradiction_severity"].iloc[i]),
            )
            for i in range(len(X_symbolic))
        ], name="triage")
