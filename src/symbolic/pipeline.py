import pandas as pd
from src.symbolic.rule_engine import RuleEngine
from src.symbolic.conflict import ConflictAnalyzer
from src.symbolic.fsm import DiagnosticFSM

SYMBOLIC_FEATURE_NAMES = [
    "certainty_psoriasis",
    "certainty_seborrheic_dermatitis",
    "certainty_lichen_planus",
    "certainty_pityriasis_rosea",
    "certainty_chronic_dermatitis",
    "certainty_pityriasis_rubra_pilaris",
    "conflict_load",
    "contradiction_severity",
    "fsm_state",
]


class SymbolicPipeline:
    def __init__(self, rules_dir: str):
        self._rule_engine = RuleEngine(rules_dir)
        self._conflict_analyzer = ConflictAnalyzer()
        self._fsm = DiagnosticFSM()

    def _process_row(self, row: pd.Series) -> dict:
        certainty_scores = self._rule_engine.fire(row)
        conflict_load, contradiction_severity = self._conflict_analyzer.analyze(certainty_scores)
        fsm_state = self._fsm.traverse(certainty_scores, conflict_load)
        return {
            "certainty_psoriasis": certainty_scores["psoriasis"],
            "certainty_seborrheic_dermatitis": certainty_scores["seborrheic_dermatitis"],
            "certainty_lichen_planus": certainty_scores["lichen_planus"],
            "certainty_pityriasis_rosea": certainty_scores["pityriasis_rosea"],
            "certainty_chronic_dermatitis": certainty_scores["chronic_dermatitis"],
            "certainty_pityriasis_rubra_pilaris": certainty_scores["pityriasis_rubra_pilaris"],
            "conflict_load": conflict_load,
            "contradiction_severity": contradiction_severity,
            "fsm_state": fsm_state,
        }

    def transform(self, X_fuzzy: pd.DataFrame) -> pd.DataFrame:
        rows = [self._process_row(row) for _, row in X_fuzzy.iterrows()]
        return pd.DataFrame(rows, columns=SYMBOLIC_FEATURE_NAMES).reset_index(drop=True)

    def explain(self, row: pd.Series) -> dict:
        certainty_scores = self._rule_engine.fire(row)
        conflict_load, contradiction_severity = self._conflict_analyzer.analyze(certainty_scores)
        fsm_state = self._fsm.traverse(certainty_scores, conflict_load)
        fired_rules = self._rule_engine.get_fired_rules(row)
        return {
            "certainty_scores": certainty_scores,
            "conflict_load": round(conflict_load, 4),
            "contradiction_severity": round(contradiction_severity, 4),
            "fsm_state": fsm_state,
            "fired_rules": fired_rules,
        }
