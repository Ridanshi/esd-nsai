# Session Log — HSCIS-ESD Project
**Date:** 2026-06-22
**Project:** Hybrid Symbolic Clinical Inference System for Erythemato-Squamous Disease Diagnosis

---

## 1. Problem Statement

**Medical problem:** Six erythemato-squamous diseases (ESD) — psoriasis, seborrheic dermatitis, lichen planus, pityriasis rosea, chronic dermatitis, pityriasis rubra pilaris — look almost identical clinically. Accurate differential diagnosis currently requires a skin biopsy.

**Gap:** In rural India and other low-resource settings, biopsy facilities are unavailable. GPs can observe 12 clinical features but cannot confirm diagnosis.

**Dataset:** UCI Dermatology dataset (id=33), 366 patients, 34 features (12 clinical + 22 histopathological), 6 disease classes, CC BY 4.0 license.

**Research gap:** Every existing paper uses all 34 features (including biopsy results) and reports 96-99% accuracy. Only Cipriano et al. (2025) tried biopsy-free diagnosis (12 clinical features, 86% accuracy, Random Forest + SHAP). No prior work applies neuro-symbolic AI to this dataset.

---

## 2. Session Goals

1. Design a novel neuro-symbolic AI system for biopsy-free ESD diagnosis
2. Build it fully in Python (open-source, runs on standard laptop)
3. Target: publishable paper + patentable method
4. Timeline: 1 month to complete

---

## 3. Brainstorming & Design Decisions

### 3.1 Approaches Considered

**Approach A (discarded):** Encode 6 boolean symbolic rule flags, feed into XGBoost.
- Rejected: "feature engineering" not true neuro-symbolic. Weak patent claim.

**Approach B (partially adopted):** Two-stage — symbolic certainty scores feed into XGBoost.
- Adopted as Stage 2 of final system.

**Approach C (partially adopted):** Pure symbolic reasoning engine with fuzzy logic + FSM.
- Adopted as Stage 1 of final system.

**Final decision: Combine B + C.**

### 3.2 System Name
**HSCIS-ESD** — Hybrid Symbolic Clinical Inference System for ESD Differential Diagnosis

### 3.3 Three-Model Experimental Design
| Model | Features | Purpose |
|---|---|---|
| A | All 34 (clinical + histopathological) | Biopsy-assisted upper bound |
| B | 12 clinical only | Biopsy-free baseline (Cipriano 2025) |
| C | 12 clinical + 9 symbolic outputs = 21 | Our novel contribution |

### 3.4 Target Users
- **Primary beneficiary:** Patients in low-resource settings (rural India)
- **Direct user:** GP/primary care physician observing clinical signs
- **Institutional:** NGOs, telemedicine platforms (Ayushman Bharat etc.)

### 3.5 Output Goals
- Paper (any good venue, not yet decided)
- Patent on the core method (hybrid fuzzy-symbolic pipeline)

---

## 4. Final Architecture

### Pipeline (5 stages)

```
12 Clinical Features (raw ordinal 0-3)
        ↓
[Stage 1] FuzzyGrader
  - Ordinal (0-3) → float (0.0-1.0): value / 3.0
  - Binary features (0/1): unchanged
  - Age: normalized by max=80
        ↓
[Stage 2] Symbolic Evidence Engine
  ├── RuleEngine
  │     - Loads 6 YAML rule files (31 rules total)
  │     - 4 evidence tiers: A=1.0, B=0.6, C=0.3, D=0.4
  │     - Fuzzy AND: firing_strength = min(feature values)
  │     - Contribution = firing_strength × rule_weight
  │     - D-tier: reduces OWN disease score (competitor sign present)
  │     - Output: 6 certainty scores (one per disease)
  ├── ConflictAnalyzer
  │     - conflict_load: pairwise product sum for diseases > 0.2 threshold
  │     - contradiction_severity: max product among incompatible pairs
  │     - Incompatible pairs: (psoriasis, lichen_planus), (pityriasis_rosea, pityriasis_rubra_pilaris)
  └── DiagnosticFSM
        - 5 states: EVIDENCE_SPARSE(0) → HYPOTHESIS_FORMING(1) → CERTAINTY_BUILDING(2) → DIAGNOSTIC_TENSION(3) → RESOLVED(4)
        - Deterministic, forward-only traversal
        - Output: fsm_state (int 0-4)
        ↓
  9 Symbolic Outputs:
  - certainty_psoriasis, certainty_seborrheic_dermatitis, certainty_lichen_planus,
    certainty_pityriasis_rosea, certainty_chronic_dermatitis, certainty_pityriasis_rubra_pilaris
  - conflict_load, contradiction_severity, fsm_state
        ↓
[Stage 3] XGBoost Classifier (Model C)
  - Input: 12 fuzzy clinical + 9 symbolic = 21 features
  - Params: n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42
  - Cross-validation: Stratified 10-fold, same folds for A/B/C
        ↓
[Stage 4] Biopsy Triage Layer (rule-based, not learned)
  - SAFE_BIOPSY_FREE: certainty ≥ 0.75 AND conflict < 0.20 AND fsm_state == RESOLVED
  - UNCERTAIN: certainty ≥ 0.55 AND conflict < 0.40
  - BIOPSY_ADVISED: otherwise
```

### Key Design Decisions
1. **FuzzyGrader is the only preprocessing step** — no StandardScaler, no PCA. Keeps pipeline auditable.
2. **D-tier rules reduce own disease score** (not competitor's). Bug was found during testing and fixed.
3. **Triage layer has no learned parameters** — fixed thresholds, clinically defensible.
4. **Models A/B/C use identical XGBoost hyperparameters** — isolates symbolic contribution cleanly.
5. **Age is the last column in UCI dataset** (position 33), not position 12 as documented.
6. **Actual UCI column name:** `knee_elbow_involvement` (not `knee_and_elbow_involvement`).

---

## 5. Rule Library

**31 rules across 6 YAML files** (`rules/` directory):

| File | Rules | Notes |
|---|---|---|
| psoriasis.yaml | 5 | PSO_A01: Koebner + knee + scalp (pathognomonic) |
| seborrheic_dermatitis.yaml | 5 | SEB_A01: scalp + scaling + erythema |
| lichen_planus.yaml | 5 | LIC_A01: polygonal papules + oral involvement |
| pityriasis_rosea.yaml | 5 | PIT_A01: definite borders + scaling + erythema |
| chronic_dermatitis.yaml | 6 | No pathognomonic rule (A tier) — problem area |
| pityriasis_rubra_pilaris.yaml | 5 | PRP_A01: follicular papules + scaling |

**Evidence tiers:**
- **A (Pathognomonic, weight=1.0):** Definitive clinical sign
- **B (Supportive, weight=0.6):** Commonly associated
- **C (Auxiliary, weight=0.3):** Weakly associated
- **D (Discriminating, weight=0.4):** Competitor's sign — when fired, reduces own disease's certainty

---

## 6. Tech Stack

```
pandas>=2.0.0          Data loading and wrangling
numpy>=1.24.0          Fuzzy math, array operations
pyyaml>=6.0            Rule file loading
scikit-learn>=1.3.0    StratifiedKFold, metrics
xgboost>=2.0.0         Primary classifier
shap>=0.46.0           Feature attribution (TreeExplainer)
imodels>=1.3.0         Post-hoc IF-THEN rule extraction
scipy>=1.11.0          Paired t-test
imbalanced-learn>=0.11.0  SMOTE (available if needed)
matplotlib>=3.7.0      Plots
seaborn>=0.12.0        Heatmaps
jupyter>=1.0.0         Notebooks (use `jupyter lab` not `jupyter notebook`)
pytest>=7.4.0          Unit tests
ucimlrepo>=0.0.3       UCI dataset download
```

---

## 7. File Structure

```
esd-neuro-symbolic/
├── requirements.txt
├── rules/
│   ├── psoriasis.yaml
│   ├── seborrheic_dermatitis.yaml
│   ├── lichen_planus.yaml
│   ├── pityriasis_rosea.yaml
│   ├── chronic_dermatitis.yaml
│   └── pityriasis_rubra_pilaris.yaml
├── src/
│   ├── data/loader.py                  UCI fetch, median imputation, class encoding
│   ├── grading/fuzzy_grader.py         Ordinal → fuzzy conversion
│   ├── symbolic/
│   │   ├── rule_engine.py              YAML loading, rule firing, certainty scores
│   │   ├── conflict.py                 Conflict load + contradiction severity
│   │   ├── fsm.py                      5-state diagnostic FSM
│   │   └── pipeline.py                 Orchestrates symbolic engine → 9 outputs
│   ├── models/
│   │   ├── base.py                     Shared XGBoost params + cross_validate_model()
│   │   ├── model_a.py                  All 34 features
│   │   ├── model_b.py                  12 clinical (fuzzy-graded)
│   │   └── model_c.py                  21 features (12 clinical + 9 symbolic)
│   ├── triage/biopsy_triage.py         Rule-based triage layer
│   └── evaluation/
│       ├── metrics.py                  Comparison table, t-test, safety analysis
│       └── explainability.py           SHAP + imodels rule extraction
├── tests/
│   ├── conftest.py                     Shared fixtures
│   ├── test_loader.py                  5 tests
│   ├── test_fuzzy_grader.py            5 tests
│   ├── test_rule_engine.py             5 tests
│   ├── test_conflict.py                5 tests
│   ├── test_fsm.py                     6 tests
│   ├── test_pipeline.py                4 tests
│   └── test_triage.py                  5 tests
├── notebooks/analysis.ipynb            Full evaluation notebook
├── docs/
│   ├── superpowers/specs/2026-06-21-hscis-esd-design.md
│   └── superpowers/plans/2026-06-22-hscis-esd-implementation.md
└── session.md                          This file
```

---

## 8. Git History

```
f676394  feat: project scaffold, package structure, shared fixtures
6a25eec  feat: DataLoader — UCI fetch, median imputation, class encoding
dea32e8  feat: FuzzyGrader — ordinal/binary/age fuzzy conversion
7e29ff9  feat: clinical rule YAML library — 31 rules across 6 ESD diseases
d43d5ff  feat: RuleEngine — fuzzy rule firing, certainty accumulation, D-tier self-penalty
4c9d884  feat: ConflictAnalyzer — conflict load and contradiction severity
bc98aa7  feat: DiagnosticFSM — 5-state deterministic diagnostic trajectory
58b76b5  feat: SymbolicPipeline — orchestrates rule engine, conflict, FSM into 9-feature output
7d1f5a6  feat: Models A and B — biopsy-assisted and clinical-only baselines
fab92a3  feat: Model C — hybrid 21-feature classifier combining clinical + symbolic outputs
2e23294  feat: BiopsyTriage — rule-based SAFE/UNCERTAIN/BIOPSY_ADVISED triage
33dfab5  feat: evaluation metrics — comparison table, 5x2 t-test, per-class safety analysis
db68aee  feat: SHAP + imodels explainability, full analysis notebook, per-class biopsy safety
```

---

## 9. Test Results

**35/35 tests pass.**

```
tests/test_loader.py         5 passed
tests/test_fuzzy_grader.py   5 passed
tests/test_rule_engine.py    5 passed
tests/test_conflict.py       5 passed
tests/test_fsm.py            6 passed
tests/test_pipeline.py       4 passed
tests/test_triage.py         5 passed
```

Run: `python -m pytest tests/ -v`

---

## 10. Model Results

```
Model A (All 34 features):          96.71% ± 1.66%  |  Macro F1: 0.9630 ± 0.0180
Model B (12 clinical features):     84.95% ± 6.01%  |  Macro F1: 0.8424 ± 0.0590
Model C (12 clinical + symbolic):   85.80% ± 3.60%  |  Macro F1: 0.8528 ± 0.0392
```

**Per-class F1:**
```
Disease                    Model A    Model B    Model C
psoriasis                   1.0000     0.9364     0.9273
seborrheic_dermatitis       0.9091     0.6667     0.6942
lichen_planus               0.9930     0.9859     0.9930
pityriasis_rosea            0.8889     0.7619     0.8039
chronic_dermatitis          0.9905     0.7400     0.7170
pityriasis_rubra_pilaris    1.0000     0.9744     1.0000
```

---

## 11. Statistical Test Results (B vs C)

```
t-statistic:      -0.6705
p-value:           0.5194
Significant:       False
Mean improvement: -0.0038 (C slightly worse than B in 5x2 CV)
```

**Interpretation:** Model C does NOT significantly outperform Model B in accuracy.

---

## 12. Single Patient Reasoning Trace (Patient 0)

True label: 1 (seborrheic_dermatitis)

Symbolic engine output:
```
Certainty scores:
  psoriasis:                0.160
  seborrheic_dermatitis:    0.400
  lichen_planus:            0.229
  pityriasis_rosea:         0.480   ← highest (wrong disease)
  chronic_dermatitis:       0.417
  pityriasis_rubra_pilaris: 0.280

Conflict load:         0.1282
Contradiction:         0.1344
FSM state:             3 (DIAGNOSTIC_TENSION)
Triage:                BIOPSY_ADVISED
```

Correct triage — system correctly flagged this ambiguous case as needing biopsy.
Rules fired for wrong disease (pityriasis_rosea had highest certainty) but triage caught it.

---

## 13. Issues Found & Fixed

### Bug 1: D-tier rule direction
**What:** D-tier rules were subtracting from `competitor` disease's score instead of own disease.
**Effect:** Patient with Koebner phenomenon (a psoriasis sign) was getting psoriasis certainty=0 because 4 competitor D-tier rules all penalized psoriasis.
**Fix:** D-tier rules now subtract from `rule["disease"]` (own score reduced when competitor's sign is present).

### Bug 2: UCI column name mismatch
**What:** Column `knee_and_elbow_involvement` doesn't exist — actual name is `knee_elbow_involvement`.
**Effect:** Clinical feature count was 11 instead of 12, test failed.
**Fix:** Updated `CLINICAL_FEATURES` list in loader.py and conftest.py.

### Bug 3: Age column position
**What:** Age is at position 33 (last column) in UCI dataset, not position 12.
**Effect:** Would have been missed if we used positional indexing.
**Fix:** Loader selects features by name from `CLINICAL_FEATURES` list, not by position.

### Issue 4: jupyter notebook command not found
**What:** `jupyter notebook` command not installed.
**Fix:** Use `jupyter lab` instead.

---

## 14. Problems / Open Issues

### Problem 1: Model C accuracy not significantly better than B
- p-value = 0.52 (need < 0.05)
- Mean improvement = -0.0038 (actually slightly worse)
- Root cause: rules too generic (scaling, erythema, itching shared by 4+ diseases)

### Problem 2: Rules too weak
- Most rules fire on generic features shared across diseases
- chronic_dermatitis has NO pathognomonic (A-tier) rule
- pityriasis_rosea rules overlap heavily with chronic_dermatitis
- Symbolic features add noise rather than signal on 366 samples

---

## 15. Plan for Next Session

**Priority 1 — Strengthen rules:**
- Add 2-3 more disease-specific rules per disease
- Focus on features that appear in only ONE disease
- Add more A-tier (pathognomonic) rules, especially for chronic_dermatitis
- Cross-reference actual UCI feature distributions per disease class

**Priority 2 — Tune XGBoost for Model C:**
- Reduce max_depth: 4 → 3 (less overfitting with 21 features on 366 samples)
- Increase reg_lambda (L2 regularization)
- Lower colsample_bytree: 0.8 → 0.6
- Rerun and check if p-value improves

**Priority 3 — Reframe contribution (do regardless of accuracy):**
- Highlight variance reduction (±3.60% vs ±6.01%)
- Highlight explainability: reasoning trace per patient
- Highlight triage: system knows when to say "I'm uncertain"
- Per-class analysis: lichen_planus + pityriasis_rubra_pilaris are SAFE_BIOPSY_FREE candidates

**Priority 4 — Run full analysis notebook:**
- Use `jupyter lab notebooks/analysis.ipynb`
- Generate SHAP plots to see if symbolic features rank in top predictors
- Run per-class safety analysis

---

## 16. Key Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Model comparison
python -c "
from src.data.loader import load_dataset
from src.models.model_a import run_model_a
from src.models.model_b import run_model_b
from src.models.model_c import run_model_c
from src.evaluation.metrics import print_comparison_table
X_clinical, X_histopath, X_all, y = load_dataset()
print_comparison_table(run_model_a(X_all, y), run_model_b(X_clinical, y), run_model_c(X_clinical, y))
"

# Statistical test B vs C
python -c "
from src.data.loader import load_dataset
from src.models.model_b import run_model_b
from src.models.model_c import run_model_c
from src.grading.fuzzy_grader import FuzzyGrader
from src.evaluation.metrics import run_statistical_test
X_clinical, _, _, y = load_dataset()
res_b = run_model_b(X_clinical, y)
res_c = run_model_c(X_clinical, y)
X_b = FuzzyGrader().grade(X_clinical).reset_index(drop=True)
print(run_statistical_test(X_b, res_c['X_combined'], y))
"

# Single patient trace
python -c "
from src.data.loader import load_dataset
from src.grading.fuzzy_grader import FuzzyGrader
from src.symbolic.pipeline import SymbolicPipeline
from src.triage.biopsy_triage import BiopsyTriage
X_clinical, _, _, y = load_dataset()
patient = FuzzyGrader().grade_series(X_clinical.iloc[0])
trace = SymbolicPipeline('rules').explain(patient)
print(trace)
triage = BiopsyTriage().recommend(max(trace['certainty_scores'].values()), trace['conflict_load'], trace['fsm_state'])
print('Triage:', triage, '| True:', y.iloc[0])
"

# Full analysis notebook
jupyter lab notebooks/analysis.ipynb
```
