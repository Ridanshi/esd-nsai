import numpy as np

INCOMPATIBLE_PAIRS = [
    ("psoriasis", "lichen_planus"),
    ("pityriasis_rosea", "pityriasis_rubra_pilaris"),
]

CONFLICT_THRESHOLD = 0.2


class ConflictAnalyzer:
    def analyze(self, certainty_scores: dict) -> tuple[float, float]:
        """
        Returns (conflict_load, contradiction_severity), both in [0.0, 1.0].

        conflict_load: normalized sum of pairwise products for diseases above threshold.
        contradiction_severity: max pairwise product among clinically incompatible pairs.
        """
        diseases = list(certainty_scores.keys())
        scores = [certainty_scores[d] for d in diseases]

        active = [(d, s) for d, s in zip(diseases, scores) if s > CONFLICT_THRESHOLD]
        conflict_sum = 0.0
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                conflict_sum += active[i][1] * active[j][1]

        n = len(active)
        max_pairs = n * (n - 1) / 2 if n > 1 else 1
        conflict_load = float(np.clip(conflict_sum / max_pairs, 0.0, 1.0)) if max_pairs > 0 else 0.0

        contradiction_severity = 0.0
        for d1, d2 in INCOMPATIBLE_PAIRS:
            s1 = certainty_scores.get(d1, 0.0)
            s2 = certainty_scores.get(d2, 0.0)
            contradiction_severity = max(contradiction_severity, s1 * s2)

        contradiction_severity = float(np.clip(contradiction_severity, 0.0, 1.0))
        return conflict_load, contradiction_severity
