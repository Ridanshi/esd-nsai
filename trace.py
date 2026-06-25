from src.data.loader import load_dataset
from src.grading.fuzzy_grader import FuzzyGrader
from src.symbolic.pipeline import SymbolicPipeline
from src.triage.biopsy_triage import BiopsyTriage

X_clinical, _, _, y = load_dataset()
grader = FuzzyGrader()
pipeline = SymbolicPipeline("rules")
triage = BiopsyTriage()

for i in range(10):
    patient = grader.grade_series(X_clinical.iloc[i])
    trace = pipeline.explain(patient)
    scores = trace["certainty_scores"]
    top = max(scores, key=scores.get)
    top_score = scores[top]
    rec = triage.recommend(top_score, trace["conflict_load"], trace["fsm_state"])
    correct = "✓" if top == list(scores.keys())[y.iloc[i]] else "✗"
    print(
        f"[{i}] true={y.iloc[i]} "
        f"top={top}({top_score:.2f}) "
        f"conflict={trace['conflict_load']:.2f} "
        f"fsm={trace['fsm_state']} "
        f"→ {rec} {correct}"
    )
    print("  scores:", {k: round(v, 2) for k, v in scores.items()})
