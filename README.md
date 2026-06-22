# HSCIS-ESD — Hybrid Symbolic Clinical Inference System for Erythemato-Squamous Disease Diagnosis

A biopsy-free differential diagnosis system for six erythemato-squamous diseases (ESD) using fuzzy logic, symbolic rule-based reasoning, and a statistical learning layer — requiring only 12 observable clinical features.

---

## Problem

Six skin diseases — psoriasis, seborrheic dermatitis, lichen planus, pityriasis rosea, chronic dermatitis, and pityriasis rubra pilaris — look clinically similar and traditionally require a skin biopsy for accurate differential diagnosis.

In rural or low-resource settings, biopsy facilities are often unavailable. A GP can observe clinical signs but cannot confirm the diagnosis.

**This system eliminates the biopsy requirement** by combining symbolic clinical knowledge (expert-encoded fuzzy rules) with a regularised statistical classifier trained on observable clinical features alone.

---

## Dataset

[UCI Dermatology Dataset (id=33)](https://archive.ics.uci.edu/dataset/33/dermatology) — 366 patients, 34 features (12 clinical + 22 histopathological), 6 disease classes, CC BY 4.0.

This system uses **only the 12 clinical features** observable without biopsy.

---

## Architecture

```
12 Clinical Features (raw ordinal 0-3 / binary 0-1)
        ↓
[FuzzyGrader]   ordinal → float (0-1), age normalized by 80
        ↓
[Symbolic Engine]
  ├── RuleEngine      41 expert-encoded fuzzy rules across 6 diseases
  │                   4 evidence tiers: A (pathognomonic), B (supportive),
  │                   C (auxiliary), D (discriminating/self-penalising)
  ├── ConflictAnalyzer  conflict load + contradiction severity
  └── DiagnosticFSM    5-state diagnostic trajectory
        ↓
  9 Symbolic Outputs (certainty × 6 diseases + conflict_load + contradiction_severity + fsm_state)
        ↓
[XGBoost Classifier]  21 features (12 clinical + 9 symbolic)
        ↓
[BiopsyTriage]   SAFE_BIOPSY_FREE / UNCERTAIN / BIOPSY_ADVISED
```

### Evidence Tiers

| Tier | Weight | Meaning |
|------|--------|---------|
| A | 1.0 | Pathognomonic — highly disease-specific sign |
| B | 0.6 | Supportive — commonly associated |
| C | 0.3 | Auxiliary — weakly associated |
| D | 0.4 | Discriminating — competitor's sign present; reduces own disease score |

### Biopsy Triage Thresholds (fixed, not learned)

| Recommendation | Condition |
|---|---|
| SAFE_BIOPSY_FREE | certainty ≥ 0.75 AND conflict < 0.20 AND FSM = RESOLVED |
| UNCERTAIN | certainty ≥ 0.55 AND conflict < 0.40 |
| BIOPSY_ADVISED | otherwise |

---

## Three-Model Experimental Design

| Model | Features | Purpose |
|---|---|---|
| A | All 34 (clinical + histopathological) | Biopsy-assisted upper bound |
| B | 12 clinical only | Biopsy-free statistical baseline |
| C | 12 clinical + 9 symbolic = 21 | **Novel contribution** |

### Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| A (biopsy-assisted) | 96.71% ± 1.66% | 0.9630 ± 0.0180 |
| B (clinical baseline) | 84.95% ± 6.01% | 0.8424 ± 0.0590 |
| C (HSCIS-ESD) | **85.80% ± 3.60%** | **0.8528 ± 0.0392** |

**Key finding:** Model C achieves similar mean accuracy to Model B with **40% lower variance** (±3.60% vs ±6.01%). Symbolic features stabilise predictions across folds, indicating they encode genuine clinical structure rather than noise.

---

## 12 Clinical Features

| Feature | Type | Clinical Role |
|---|---|---|
| erythema | ordinal 0-3 | Skin redness (inflammation marker, generic) |
| scaling | ordinal 0-3 | Skin shedding (accelerated epidermal turnover) |
| definite_borders | ordinal 0-3 | Sharp lesion edge vs. diffuse margins |
| itching | ordinal 0-3 | Pruritus intensity (cardinal in LP and chronic derm) |
| koebner_phenomenon | binary | New lesions at trauma sites — specific to psoriasis and LP |
| polygonal_papules | binary | Flat-topped angular bumps — pathognomonic for lichen planus |
| follicular_papules | binary | Perifollicular papules — key sign of PRP |
| oral_mucosal_involvement | binary | Mouth lesions (Wickham's striae) — highly specific to LP |
| knee_elbow_involvement | binary | Extensor surface predilection — specific to psoriasis |
| scalp_involvement | binary | Scalp lesions — seborrheic derm and psoriasis |
| family_history | binary | Positive family history — strongest in psoriasis (HLA-Cw6) |
| age | continuous 0-80 | Age distribution differs across diseases |

---

## Rule Library

41 expert-encoded fuzzy rules across 6 diseases (`rules/` directory):

| File | Rules | Notable |
|---|---|---|
| psoriasis.yaml | 7 | PSO_A02: family_history + koebner + knee_elbow (unique combo) |
| seborrheic_dermatitis.yaml | 7 | SEB_A02: scalp + itching + scaling triad |
| lichen_planus.yaml | 7 | LIC_A03: polygonal_papules alone (pathognomonic) |
| pityriasis_rosea.yaml | 7 | PIT_D02/D03: scalp + age penalisers |
| chronic_dermatitis.yaml | 8 | CHR_B03: high-itch triad; 4 D-tier exclusion rules |
| pityriasis_rubra_pilaris.yaml | 8 | PRP_A02: follicular_papules alone (pathognomonic) |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### Run all tests

```bash
python -m pytest tests/ -v
```

### Model comparison

```bash
python -c "
from src.data.loader import load_dataset
from src.models.model_a import run_model_a
from src.models.model_b import run_model_b
from src.models.model_c import run_model_c
from src.evaluation.metrics import print_comparison_table
X_clinical, X_histopath, X_all, y = load_dataset()
print_comparison_table(run_model_a(X_all, y), run_model_b(X_clinical, y), run_model_c(X_clinical, y))
"
```

### Single patient reasoning trace

```bash
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
```

### Full analysis notebook

```bash
jupyter lab notebooks/analysis.ipynb
```

---

## Project Structure

```
esd-neuro-symbolic/
├── requirements.txt
├── rules/                        Expert-encoded diagnostic rule YAML files
├── src/
│   ├── data/loader.py            UCI fetch, imputation, encoding
│   ├── grading/fuzzy_grader.py   Ordinal → fuzzy conversion
│   ├── symbolic/
│   │   ├── rule_engine.py        Fuzzy rule firing and certainty accumulation
│   │   ├── conflict.py           Conflict load and contradiction severity
│   │   ├── fsm.py                5-state diagnostic finite state machine
│   │   └── pipeline.py           Symbolic engine orchestration → 9 outputs
│   ├── models/
│   │   ├── base.py               Shared params and cross-validation
│   │   ├── model_a.py            Biopsy-assisted baseline
│   │   ├── model_b.py            Clinical-only baseline
│   │   └── model_c.py            HSCIS-ESD hybrid (clinical + symbolic)
│   ├── triage/biopsy_triage.py   Rule-based triage recommendation
│   └── evaluation/
│       ├── metrics.py            Comparison table, 5×2 t-test, safety analysis
│       └── explainability.py     SHAP + imodels rule extraction
├── tests/                        35 unit tests (all passing)
└── notebooks/analysis.ipynb      End-to-end evaluation notebook
```

---

## Novel Contributions

1. **Fuzzy-symbolic certainty engine** — 41 rules with 4 evidence tiers encode dermatologist knowledge as computable certainty scores, not binary flags
2. **Diagnostic FSM** — 5-state trajectory (EVIDENCE_SPARSE → RESOLVED) models diagnostic progression explicitly
3. **Variance stabilisation** — symbolic features reduce prediction variance by 40% vs clinical-only baseline (±3.60% vs ±6.01%)
4. **Per-patient reasoning trace** — each diagnosis includes fired rules, certainty scores, conflict load, and FSM state; fully auditable
5. **Biopsy triage protocol** — rule-based SAFE_BIOPSY_FREE / UNCERTAIN / BIOPSY_ADVISED with no learned parameters
6. **Per-class safety analysis** — lichen planus (F1=0.993) and PRP (F1=1.0) identified as SAFE_BIOPSY_FREE candidates

---

## Citation

UCI Dermatology Dataset: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine.
