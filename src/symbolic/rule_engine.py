import os
import glob
import yaml
import numpy as np
import pandas as pd

DISEASES = [
    "psoriasis", "seborrheic_dermatitis", "lichen_planus",
    "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris"
]


class RuleEngine:
    def __init__(self, rules_dir: str):
        self._rules: list[dict] = []
        self._max_weight: dict[str, float] = {d: 0.0 for d in DISEASES}
        self._load_rules(rules_dir)

    def _load_rules(self, rules_dir: str) -> None:
        for path in glob.glob(os.path.join(rules_dir, "*.yaml")):
            with open(path) as f:
                rules = yaml.safe_load(f)
            for rule in rules:
                self._rules.append(rule)
                disease = rule["disease"]
                if rule.get("tier") != "D":
                    self._max_weight[disease] = (
                        self._max_weight.get(disease, 0.0) + rule["weight"]
                    )

    def _fire_rule(self, rule: dict, patient: pd.Series) -> float:
        """Returns firing strength (0.0 if any condition unmet)."""
        strengths = []
        for cond in rule["conditions"]:
            feat = cond["feature"]
            threshold = cond["threshold"]
            value = float(patient.get(feat, 0.0))
            if value < threshold:
                return 0.0
            strengths.append(value)
        return float(np.min(strengths)) if strengths else 0.0

    def fire(self, patient: pd.Series) -> dict[str, float]:
        """Returns certainty score per disease, normalized to [0, 1]."""
        accumulated = {d: 0.0 for d in DISEASES}
        competitor_penalties = {d: 0.0 for d in DISEASES}

        for rule in self._rules:
            strength = self._fire_rule(rule, patient)
            if strength == 0.0:
                continue
            disease = rule["disease"]
            if rule.get("tier") == "D":
                # D-tier: competitor's sign present → penalise this disease's score
                competitor_penalties[disease] += rule["weight"] * strength
            else:
                accumulated[disease] += rule["weight"] * strength

        scores = {}
        for d in DISEASES:
            max_w = self._max_weight.get(d, 1.0)
            if max_w == 0.0:
                scores[d] = 0.0
            else:
                raw = accumulated[d] - competitor_penalties.get(d, 0.0)
                scores[d] = float(np.clip(raw / max_w, 0.0, 1.0))
        return scores

    def get_fired_rules(self, patient: pd.Series) -> list[dict]:
        """Returns metadata for all rules that fired (strength > 0)."""
        fired = []
        for rule in self._rules:
            strength = self._fire_rule(rule, patient)
            if strength > 0.0:
                fired.append({
                    "id": rule["id"],
                    "disease": rule["disease"],
                    "tier": rule.get("tier"),
                    "firing_strength": round(strength, 4),
                    "contribution": round(strength * rule["weight"], 4),
                })
        return fired
