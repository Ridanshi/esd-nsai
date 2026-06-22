# HSCIS-ESD: Hybrid Symbolic Clinical Inference System for ESD Differential Diagnosis
**Design Specification — 2026-06-21**

---

## 1. Problem & Motivation

Erythemato-Squamous Diseases (ESD) are a group of 6 skin diseases — psoriasis, seborrheic dermatitis, lichen planus, pityriasis rosea, chronic dermatitis, and pityriasis rubra pilaris — that share overlapping clinical signs (erythema, scaling). Accurate differential diagnosis currently requires histopathological examination via skin biopsy.

In low-resource settings (rural India, sub-Saharan Africa), biopsy facilities are absent or unaffordable. This creates a diagnostic gap: GPs can observe clinical signs but cannot confirm diagnosis.

Every existing paper on the UCI Dermatology dataset uses all 34 features including biopsy results, achieving 96–99% accuracy. This is clinically meaningless — it uses biopsy output to predict biopsy output. Only Cipriano et al. (2025) attempted biopsy-free diagnosis using 12 clinical features (86% accuracy, Random Forest + SHAP), without clinical knowledge integration.

No prior work applies a hybrid symbolic-statistical approach to this dataset.

---

## 2. System Overview

**System Name:** Hybrid Symbolic Clinical Inference System for ESD (HSCIS-ESD)

**Core idea:** Encode published dermatological diagnostic criteria as a fuzzy symbolic reasoning engine. Use its outputs to augment a statistical classifier. The symbolic layer contributes structured clinical knowledge; the statistical layer handles noise and atypical presentations.

**Target user:** Primary care physician / GP in a low-resource setting with no biopsy access.

**Target beneficiary:** Patients in rural/resource-constrained settings who currently go undiagnosed.

**Deployment context:** Runs on a standard laptop, no internet, no GPU, no biopsy required.

---

## 3. Goals & Non-Goals

### Goals
1. Exceed biopsy-free clinical-only baseline (Model B ~86%) using symbolic knowledge integration
2. Identify which ESD diseases can be safely diagnosed without biopsy (per-class analysis)
3. Produce interpretable, clinically auditable reasoning traces
4. Demonstrate symbolic features contribute meaningfully (via SHAP attribution)

### Non-Goals
- Not a clinical deployment system — research prototype
- Not a replacement for dermatologists in well-equipped hospitals
- Not trained on histopathological features in Models B and C (excluded by design)
- Not a general dermatology classifier — strictly ESD differential diagnosis

### Success Criteria
| Metric | Target |
|---|---|
| Model C vs Model B accuracy | Statistically significant improvement (p < 0.05, paired t-test) |
| Per-class F1 for ≥ 2 diseases | ≥ 0.85 (SAFE_BIOPSY_FREE candidates) |
| SHAP attribution | Symbolic features appear in top-10 predictors for Model C |
| Triage calibration | BIOPSY_ADVISED cases have lower top-certainty than SAFE cases |

---

## 4. Dataset

**Source:** UCI Dermatology Dataset (id=33)
**Loader:** `ucimlrepo` Python package (`fetch_ucirepo(id=33)`)
**License:** CC BY 4.0
**Size:** 366 patients, 34 features, 6 target classes
**Feature split:**
- 12 clinical features — observable without biopsy (used in Models B and C)
- 22 histopathological features — require biopsy (used only in Model A)

**Clinical features (12):**
erythema, scaling, definite_borders, itching, koebner_phenomenon, polygonal_papules, follicular_papules, oral_mucosal_involvement, knee_and_elbow_involvement, scalp_involvement, family_history, age

**Target classes (6):**
psoriasis, seborrheic_dermatitis, lichen_planus, pityriasis_rosea, chronic_dermatitis, pityriasis_rubra_pilaris

---

## 5. Three-Model Experimental Design

| Model | Input Features | Purpose |
|---|---|---|
| **Model A** | All 34 (clinical + histopathological) | Upper bound — replicates existing literature |
| **Model B** | 12 clinical only | Lower bound — Cipriano 2025 baseline |
| **Model C** | 12 clinical + 9 symbolic outputs = 21 | Our contribution |

All three models use identical XGBoost hyperparameters and identical cross-validation setup. The only variable is input features. This cleanly isolates the contribution of the symbolic engine.

---

## 6. Architecture

```
Raw Input: 12 Clinical Features (ordinal 0–3 scale)
                    │
                    ▼
        ┌───────────────────────┐
        │    Fuzzy Grader       │   ordinal → float (0.0–1.0)
        └───────────┬───────────┘
                    │  12 fuzzy features
                    ▼
        ┌───────────────────────────────────────┐
        │          Symbolic Engine              │
        │                                       │
        │  ┌─────────────────────────────────┐  │
        │  │ Rule Engine                     │  │
        │  │ - loads YAML rule files         │  │
        │  │ - fires rules (fuzzy AND logic) │  │
        │  │ - computes 6 certainty scores   │  │
        │  └──────────────┬──────────────────┘  │
        │                 │                     │
        │  ┌──────────────▼──────────────────┐  │
        │  │ Conflict Analyzer               │  │
        │  │ - conflict_load scalar          │  │
        │  │ - contradiction_severity scalar │  │
        │  └──────────────┬──────────────────┘  │
        │                 │                     │
        │  ┌──────────────▼──────────────────┐  │
        │  │ Diagnostic State Tracker (FSM)  │  │
        │  │ - 5-state FSM                   │  │
        │  │ - outputs fsm_state (int 0–4)   │  │
        │  └─────────────────────────────────┘  │
        └───────────────────┬───────────────────┘
                            │  9 symbolic outputs
                            ▼
        ┌───────────────────────────────────────┐
        │   Statistical Refinement (XGBoost)    │
        │   Input: 12 fuzzy + 9 symbolic = 21   │
        │   Output: disease label + class probs │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │         Biopsy Triage Layer           │
        │   Input: top_certainty, conflict_load,│
        │           fsm_state                   │
        │   Output: SAFE_BIOPSY_FREE /          │
        │           UNCERTAIN /                 │
        │           BIOPSY_ADVISED              │
        └───────────────────────────────────────┘
```

---

## 7. Component Specifications

### 7.1 Data Loader (`src/data/loader.py`)

**Responsibility:** Fetch UCI dataset, name features, split into clinical / histopathological subsets, handle missing values.

**Interface:**
```python
load_dataset() -> (X_clinical, X_histopath, X_all, y)
```

**Missing values:** UCI dataset has missing values in the `age` feature for some records. Strategy: median imputation (not mean — age is right-skewed in this dataset). Document imputation in paper.

**Output types:** All feature values as integers before fuzzy grading.

---

### 7.2 Fuzzy Grader (`src/grading/fuzzy_grader.py`)

**Responsibility:** Convert ordinal clinical features (0–3 scale) to fuzzy membership values (0.0–1.0).

**Transform:**
```
ordinal_value / 3.0  →  fuzzy_value
0 → 0.0, 1 → 0.33, 2 → 0.67, 3 → 1.0
```

**Binary features** (family_history, koebner_phenomenon, etc.): remain 0.0 or 1.0.

**Interface:**
```python
grade(X: pd.DataFrame) -> pd.DataFrame  # same shape, float dtype
```

---

### 7.3 Rule Engine (`src/symbolic/rule_engine.py`)

**Responsibility:** Load YAML rule files, fire rules against fuzzy features, compute per-disease certainty scores.

**Rule YAML schema:**
```yaml
- id: PSO_001
  disease: psoriasis
  tier: A                  # A=Pathognomonic, B=Supportive, C=Auxiliary, D=Discriminating
  weight: 1.0
  conditions:
    - feature: koebner_phenomenon
      threshold: 0.5
    - feature: knee_and_elbow_involvement
      threshold: 0.5
    - feature: scalp_involvement
      threshold: 0.5
```

**Evidence tier → weight:**
| Tier | Weight | Meaning |
|---|---|---|
| A | 1.0 | Pathognomonic — textbook definitive sign |
| B | 0.6 | Supportive — commonly associated |
| C | 0.3 | Auxiliary — weakly associated |
| D | 0.4 | Discriminating — when fired, subtracts 0.4 × firing_strength from the named competitor disease's certainty (not added to own disease) |

**Rule firing (AND logic):**
```
firing_strength = min(fuzzy_value for each condition)
contribution    = firing_strength × rule_weight
```

Minimum fuzzy value = weakest-link principle. An AND chain is only as strong as its least-satisfied condition.

**Certainty score per disease:**
```
certainty_d = Σ(contribution_i for fired rules of disease d)
              / Σ(weight_i for all rules of disease d)
```
Normalized to [0.0, 1.0].

**Rule counts (target):** ~5–8 rules per disease, 30–40 total across 6 YAML files.

**Interface:**
```python
fire(X_fuzzy: pd.Series) -> dict  # {disease_name: certainty_score}
```

---

### 7.4 Conflict Analyzer (`src/symbolic/conflict.py`)

**Responsibility:** Quantify diagnostic ambiguity from the certainty score distribution.

**Conflict load:** Measures simultaneous support for multiple diseases.
```
conflict_load = Σ(certainty_i × certainty_j) for all pairs i≠j where certainty > 0.2
```
High conflict_load = clinically ambiguous picture.

**Contradiction severity:** Measures co-occurrence of evidence for clinically incompatible diseases (diseases that share no pathognomonic features). Defined as max pairwise certainty product among incompatible disease pairs.

**Interface:**
```python
analyze(certainty_scores: dict) -> (conflict_load: float, contradiction_severity: float)
```

---

### 7.5 Diagnostic State Tracker (`src/symbolic/fsm.py`)

**Responsibility:** Traverse a 5-state FSM based on certainty and conflict signals. Output encodes the reasoning trajectory, not just the endpoint.

**States:**
```
0: EVIDENCE_SPARSE        — no rule has fired with certainty > 0.1
1: HYPOTHESIS_FORMING     — at least one disease certainty > 0.1
2: CERTAINTY_BUILDING     — top disease certainty > 0.4
3: DIAGNOSTIC_TENSION     — conflict_load > 0.3 OR 2+ diseases > 0.4
4: RESOLVED               — top certainty > 0.65 AND conflict_load < 0.25
                            OR top certainty > 0.80 regardless of conflict
```

**Transitions are deterministic and sequential** — a patient can only advance forward through states, never backward. Final state is the output.

**Why this matters:** Two patients with identical final certainty scores but different trajectories (one passed through DIAGNOSTIC_TENSION, one did not) are clinically different. The FSM captures this distinction.

**Interface:**
```python
traverse(certainty_scores: dict, conflict_load: float) -> int  # state 0-4
```

---

### 7.6 Symbolic Engine Outputs (9 features per patient)

```
certainty_psoriasis                (float 0–1)
certainty_seborrheic_dermatitis    (float 0–1)
certainty_lichen_planus            (float 0–1)
certainty_pityriasis_rosea         (float 0–1)
certainty_chronic_dermatitis       (float 0–1)
certainty_pityriasis_rubra_pilaris (float 0–1)
conflict_load                      (float 0–1)
contradiction_severity             (float 0–1)
fsm_state                          (int 0–4)
```

---

### 7.7 Statistical Refinement — XGBoost (`src/models/model_c.py`)

**Input:** 21 features = 12 fuzzy clinical + 9 symbolic outputs

**Classifier:** XGBoost (`xgboost.XGBClassifier`)

**Hyperparameters (shared across Models A, B, C):**
```python
XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
```

**Cross-validation:** Stratified 10-fold (`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`). Stratification required — 6 imbalanced classes on 366 samples.

**Output:** Predicted disease label + probability vector over 6 classes.

---

### 7.8 Biopsy Triage Layer (`src/triage/biopsy_triage.py`)

**Input:** top_certainty (max of 6 certainty scores), conflict_load, fsm_state

**Logic (rule-based, no learned parameters):**
```python
if top_certainty >= 0.75 and conflict_load < 0.20 and fsm_state == 4:
    return "SAFE_BIOPSY_FREE"
elif top_certainty >= 0.55 and conflict_load < 0.40:
    return "UNCERTAIN"
else:
    return "BIOPSY_ADVISED"
```

**Threshold rationale:**
- 0.75 certainty threshold: conservative — requires strong symbolic evidence before declaring safe
- 0.20 conflict threshold: allows mild overlap, blocks moderate ambiguity
- FSM state == RESOLVED (4): ensures clean reasoning trajectory, not just high endpoint certainty

**Safety property — escalation only:** Triage can only move toward BIOPSY_ADVISED as evidence weakens, never away from it. A case once flagged BIOPSY_ADVISED cannot be downgraded by any single feature.

---

## 8. Rule Library Design

**6 YAML files** — one per disease — stored in `rules/`.

**Rule derivation sources:**
- Andrews' Diseases of the Skin (clinical reference)
- Fitzpatrick's Dermatology (standard textbook)
- Published clinical diagnostic criteria for each ESD

**Per-disease rule targets:**

| Disease | Pathognomonic rules | Supportive rules | Auxiliary rules | Discriminating rules |
|---|---|---|---|---|
| Psoriasis | 1–2 | 2–3 | 1–2 | 1 |
| Seborrheic dermatitis | 1 | 2–3 | 1–2 | 1 |
| Lichen planus | 1–2 | 2 | 1 | 1 |
| Pityriasis rosea | 1 | 2 | 1–2 | 1 |
| Chronic dermatitis | 0–1 | 2–3 | 2 | 1 |
| Pityriasis rubra pilaris | 1 | 2 | 1 | 1 |

**Example rule (psoriasis):**
```yaml
# rules/psoriasis.yaml
- id: PSO_001
  disease: psoriasis
  tier: A
  weight: 1.0
  conditions:
    - feature: koebner_phenomenon
      threshold: 0.5
    - feature: knee_and_elbow_involvement
      threshold: 0.5
    - feature: scalp_involvement
      threshold: 0.5

- id: PSO_002
  disease: psoriasis
  tier: B
  weight: 0.6
  conditions:
    - feature: scaling
      threshold: 0.5
    - feature: erythema
      threshold: 0.5

- id: PSO_003
  disease: psoriasis
  tier: B
  weight: 0.6
  conditions:
    - feature: family_history
      threshold: 0.5
    - feature: itching
      threshold: 0.33
```

---

## 9. Evaluation Protocol

### Cross-Validation
- Stratified 10-fold, same folds for Models A, B, C
- Report: mean accuracy ± std across folds

### Metrics (per model)
- Overall accuracy
- Macro F1 (handles class imbalance)
- Per-class precision, recall, F1
- Confusion matrix

### Statistical Significance
- Paired t-test (5×2 cross-validation method) between Model B and Model C
- Report p-value and effect size (Cohen's d)
- Claim improvement only if p < 0.05

### Symbolic Contribution Analysis
- SHAP TreeExplainer on Model C
- Identify rank of symbolic features vs raw clinical features
- Beeswarm plot: shows which features drive which diseases

### Biopsy Triage Validation
- Per triage tier: report mean top_certainty and mean correct-classification rate
- BIOPSY_ADVISED cases should have lower accuracy than SAFE cases — validates the triage signal

### Per-class Biopsy Safety Analysis
- For each disease: what % of correctly classified cases are in SAFE_BIOPSY_FREE tier?
- This produces the paper's Table 3: "Diseases diagnosable without biopsy"

---

## 10. Interpretability Layer

### SHAP (Model-level)
- `shap.TreeExplainer(model_c)`
- Global: beeswarm plot of SHAP values — shows which of 21 features matter most
- Per-class: SHAP force plots for representative patients

### imodels (Rule Extraction)
- Fit `RuleFit` or `FIGS` from `imodels` on Model C
- Extract top IF-THEN rules
- Validate extracted rules against hand-crafted YAML rules — agreement = validation signal
- **Important:** imodels rules are post-hoc explanations of Model C. They are distinct from the YAML rules that feed into the symbolic engine. Do not conflate in paper.

### Reasoning Trace (Symbolic)
- For each patient: log which rules fired, at what strength, certainty scores per disease, FSM state traversal, final triage
- This is the system's "clinical narrative" — auditable by a physician

---

## 11. Project Structure

```
esd-neuro-symbolic/
├── data/
│   └── raw/                        # auto-downloaded by DataLoader
├── rules/
│   ├── psoriasis.yaml
│   ├── seborrheic_dermatitis.yaml
│   ├── lichen_planus.yaml
│   ├── pityriasis_rosea.yaml
│   ├── chronic_dermatitis.yaml
│   └── pityriasis_rubra_pilaris.yaml
├── src/
│   ├── data/
│   │   └── loader.py
│   ├── grading/
│   │   └── fuzzy_grader.py
│   ├── symbolic/
│   │   ├── rule_engine.py
│   │   ├── conflict.py
│   │   └── fsm.py
│   ├── models/
│   │   ├── model_a.py
│   │   ├── model_b.py
│   │   └── model_c.py
│   ├── triage/
│   │   └── biopsy_triage.py
│   └── evaluation/
│       ├── metrics.py
│       └── explainability.py
├── notebooks/
│   └── analysis.ipynb
├── tests/
│   ├── test_fuzzy_grader.py
│   ├── test_rule_engine.py
│   └── test_fsm.py
├── docs/
│   └── superpowers/specs/
│       └── 2026-06-21-hscis-esd-design.md
├── requirements.txt
└── README.md
```

---

## 12. Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, wrangling |
| `numpy` | Fuzzy math, array operations |
| `pyyaml` | Rule file loading |
| `scikit-learn` | StratifiedKFold, metrics, preprocessing |
| `xgboost` | Primary classifier (Models A, B, C) |
| `shap` | Feature attribution (TreeExplainer) |
| `imodels` | Post-hoc IF-THEN rule extraction |
| `scipy` | Paired t-test, statistical significance |
| `imbalanced-learn` | SMOTE (if class imbalance requires it) |
| `matplotlib` / `seaborn` | Plots, confusion matrices, SHAP visualizations |
| `jupyter` | Analysis notebooks |
| `pytest` | Unit tests for symbolic engine |
| `ucimlrepo` | UCI dataset download |

**Install:**
```bash
pip install pandas numpy pyyaml scikit-learn xgboost shap imodels scipy imbalanced-learn matplotlib seaborn jupyter pytest ucimlrepo
```

---

## 13. Implementation Sequence (4 Weeks)

| Week | Deliverables |
|---|---|
| 1 | `DataLoader`, `FuzzyGrader`, all 6 YAML rule files (~30–40 rules), `RuleEngine` |
| 2 | `ConflictAnalyzer`, `DiagnosticFSM`, full symbolic engine pipeline, unit tests |
| 3 | Models A, B, C with cross-validation, evaluation metrics, `BiopsyTriage` |
| 4 | SHAP analysis, imodels rule extraction, per-class safety analysis, paper draft |

---

## 14. Novel Contributions (Paper & Patent)

**Paper contributions:**
1. First application of hybrid symbolic-statistical inference to ESD biopsy-free diagnosis
2. Fuzzy tiered rule firing with certainty propagation as a clinical knowledge encoding method
3. FSM-based diagnostic trajectory tracking as an augmentation signal for statistical classifiers
4. Per-disease biopsy safety analysis — identifies which ESD diseases are safely diagnosable without biopsy

**Patentable method:**
The protectable claim is the specific pipeline: *fuzzy clinical feature grading → tiered symbolic rule firing with certainty propagation → conflict load and contradiction detection → FSM diagnostic state traversal → hybrid XGBoost integration → rule-based biopsy triage protocol*, applied to ESD differential diagnosis. Each of the 5 stages is individually novel; the pipeline combination is non-obvious.

---

*End of design specification.*
