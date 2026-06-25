"""
Mutual information scoring for the 9 engineered features.
Run once to decide which features to keep in FeatureEngineer.
"""
from src.data.loader import load_dataset
from src.grading.fuzzy_grader import FuzzyGrader
from src.grading.feature_engineer import FeatureEngineer, ENGINEERED_FEATURE_NAMES
from sklearn.feature_selection import mutual_info_classif
import pandas as pd

X_clinical, _, _, y = load_dataset()
grader = FuzzyGrader()
X_fuzzy = grader.grade(X_clinical).reset_index(drop=True)

engineer = FeatureEngineer()
X_eng = engineer.engineer(X_fuzzy)

mi_scores = mutual_info_classif(X_eng, y, random_state=42)
results = pd.Series(mi_scores, index=ENGINEERED_FEATURE_NAMES).sort_values(ascending=False)

print("Mutual Information scores (higher = more useful):\n")
for feat, score in results.items():
    bar = "#" * int(score * 80)
    keep = "KEEP" if score >= 0.05 else "DROP"
    print(f"  [{keep}] {feat:<28} {score:.4f}  {bar}")

print(f"\nThreshold: 0.05  |  Keep: {(results >= 0.05).sum()}  |  Drop: {(results < 0.05).sum()}")
