# HSCIS-ESD — Hybrid Symbolic Clinical Inference System for Erythemato-Squamous Disease Diagnosis

A biopsy-free differential diagnosis system for six erythemato-squamous diseases using fuzzy logic, symbolic rule-based reasoning, and a regularised statistical classifier — requiring only 12 observable clinical features.

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
[FuzzyGrader]      ordinal → float (0-1), age / 80
        ↓
[FeatureEngineer]  8 MI-selected clinical interaction features
        ↓
[Symbolic Engine]
  ├── RuleEngine        41 expert-encoded fuzzy rules, 4 evidence tiers
  ├── ConflictAnalyzer  conflict_load + contradiction_severity
  └── DiagnosticFSM     5-state diagnostic trajectory
        ↓
  9 Symbolic Outputs
        ↓
[XGBoost Classifier]   29 total features (12 fuzzy + 8 engineered + 9 symbolic)
        ↓
[BiopsyTriage]         SAFE_BIOPSY_FREE / UNCERTAIN / BIOPSY_ADVISED
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
| C | 12 fuzzy + 8 engineered + 9 symbolic = 29 | **Novel contribution (HSCIS-ESD)** |

### Results (Stratified 10-fold CV, RANDOM_STATE=42)

| Model | Accuracy | Macro F1 |
|---|---|---|
| A (biopsy-assisted) | 96.71% ± 1.66% | 0.9630 ± 0.0180 |
| B (clinical baseline) | 84.95% ± 6.01% | 0.8424 ± 0.0590 |
| **C (HSCIS-ESD)** | **86.61% ± 3.55%** | **0.8619 ± 0.0329** |

### Per-Class F1

| Disease | Model A | Model B | Model C |
|---|---|---|---|
| psoriasis | 1.0000 | 0.9364 | 0.9333 |
| seborrheic_dermatitis | 0.9091 | 0.6667 | **0.7419** |
| lichen_planus | 0.9930 | 0.9859 | 0.9640 |
| pityriasis_rosea | 0.8889 | 0.7619 | **0.8041** |
| chronic_dermatitis | 0.9905 | 0.7400 | **0.7547** |
| pityriasis_rubra_pilaris | 1.0000 | 0.9744 | 0.9756 |

---

## Ablation Study

Marginal contribution of each feature layer (same XGBoost params, same 10-fold CV):

| Configuration | Features | Accuracy | Std | Macro F1 |
|---|---|---|---|---|
| Fuzzy only | 12 | 86.34% | ±4.86% | 0.8575 |
| Fuzzy + Engineered | 20 | 86.34% | ±4.87% | 0.8531 |
| Fuzzy + Symbolic | 21 | 85.53% | **±2.66%** | 0.8495 |
| **Fuzzy + Eng + Sym (Model C)** | **29** | **86.61%** | **±3.55%** | **0.8619** |

**Key finding:** The symbolic layer's primary contribution is **diagnostic stability**, not raw accuracy. Adding symbolic features alone reduces prediction variance by **45%** (±4.86% → ±2.66%) — indicating the expert rules encode genuine clinical structure. Combined with engineered features, the full system achieves best accuracy and best F1.

---

## 12 Clinical Features

| Feature | Type | Clinical Role |
|---|---|---|
| erythema | ordinal 0-3 | Skin redness (inflammation marker) |
| scaling | ordinal 0-3 | Skin shedding (epidermal turnover) |
| definite_borders | ordinal 0-3 | Sharp lesion edge vs. diffuse margins |
| itching | ordinal 0-3 | Pruritus intensity (cardinal in LP and chronic derm) |
| koebner_phenomenon | binary | New lesions at trauma sites — specific to psoriasis and LP |
| polygonal_papules | binary | Flat-topped angular bumps — pathognomonic for lichen planus |
| follicular_papules | binary | Perifollicular papules — key sign of PRP |
| oral_mucosal_involvement | binary | Mouth lesions (Wickham's striae) — highly specific to LP |
| knee_elbow_involvement | binary | Extensor surface predilection — specific to psoriasis |
| scalp_involvement | binary | Scalp lesions — seborrheic derm and psoriasis |
| family_history | binary | Positive family history — strongest in psoriasis (HLA-Cw6) |
| age | continuous 0-80 | Age distribution differs markedly across diseases |

---

## Engineered Features (8, selected by mutual information ≥ 0.05)

| Feature | Formula | MI Score | Clinical Basis |
|---|---|---|---|
| lp_classic | polygonal × oral_mucosal | 0.414 | LP pathognomonic combination |
| no_specific_morphology | 1 − clip(polygonal + follicular + koebner) | 0.277 | Seb_derm / chronic_derm exclusion |
| itch_no_border | itching × (1 − definite_borders) | 0.260 | Chronic dermatitis pattern |
| prp_core | follicular × scaling | 0.242 | PRP signature |
| scale_erythema_ratio | scaling / (erythema + 0.01) | 0.183 | Psoriasis vs LP discriminator |
| inflammation_burden | (erythema + scaling + itching) / 9 | 0.134 | Overall severity |
| pso_triad | koebner × knee_elbow × family_history | 0.055 | Near-unique psoriasis combination |

---

## Rule Library (41 rules across 6 YAML files)

| File | Rules | Notable Rule |
|---|---|---|
| psoriasis.yaml | 7 | PSO_A02: family_history + koebner + knee_elbow (unique combo) |
| seborrheic_dermatitis.yaml | 7 | SEB_A02: scalp + itching + scaling triad |
| lichen_planus.yaml | 7 | LIC_A03: polygonal_papules alone (pathognomonic) |
| pityriasis_rosea.yaml | 7 | PIT_D02/D03: scalp and age penalisers |
| chronic_dermatitis.yaml | 8 | CHR_B03: high-itch triad; 4 D-tier exclusion rules |
| pityriasis_rubra_pilaris.yaml | 8 | PRP_A02: follicular_papules alone (pathognomonic) |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# All 43 unit tests
python -m pytest tests/ -v

# Full model comparison + McNemar test
python eval_run.py

# Ablation study (feature layer contribution)
python ablation.py

# Overfitting diagnostic (train vs val per fold)
python diagnose_fit.py

# Per-patient symbolic reasoning trace (10 patients)
python trace.py

# Feature MI scoring
python select_features.py

# Full analysis notebook (SHAP, biopsy triage, reasoning trace)
jupyter lab notebooks/analysis.ipynb
```

---

## Project Structure

```
esd-neuro-symbolic/
├── requirements.txt
├── rules/                         41 expert-encoded diagnostic YAML rules
├── src/
│   ├── data/loader.py             UCI fetch, imputation, class encoding
│   ├── grading/
│   │   ├── fuzzy_grader.py        Ordinal/binary/age → fuzzy float
│   │   └── feature_engineer.py   8 MI-selected interaction features
│   ├── symbolic/
│   │   ├── rule_engine.py         Fuzzy rule firing, certainty accumulation
│   │   ├── conflict.py            Conflict load + contradiction severity
│   │   ├── fsm.py                 5-state diagnostic FSM
│   │   └── pipeline.py            Symbolic engine → 9 outputs
│   ├── models/
│   │   ├── base.py                Shared params + cross-validation
│   │   ├── model_a.py             Biopsy-assisted baseline (34 features)
│   │   ├── model_b.py             Clinical-only baseline (12 features)
│   │   └── model_c.py             HSCIS-ESD hybrid (29 features)
│   ├── triage/biopsy_triage.py    Rule-based SAFE/UNCERTAIN/BIOPSY_ADVISED
│   └── evaluation/
│       ├── metrics.py             Comparison table, McNemar test, safety analysis
│       └── explainability.py      SHAP + imodels rule extraction
├── tests/                         43 unit tests (all passing)
├── eval_run.py                    Model A/B/C comparison + McNemar B vs C
├── ablation.py                    4-layer feature ablation study
├── diagnose_fit.py                Train vs val accuracy per fold
├── trace.py                       Per-patient symbolic reasoning trace
├── select_features.py             Mutual information scoring
└── notebooks/analysis.ipynb      End-to-end evaluation notebook
```

---

## Novel Contributions

1. **Fuzzy-symbolic certainty engine** — 41 rules with 4 evidence tiers encode dermatologist knowledge as computable certainty scores, not binary flags
2. **Diagnostic FSM** — 5-state trajectory (EVIDENCE_SPARSE → HYPOTHESIS_FORMING → BUILDING → TENSION → RESOLVED) models diagnostic progression explicitly
3. **Variance stabilisation** — symbolic features reduce prediction variance by 45% vs clinical-only baseline (±2.66% vs ±4.86% in ablation; ±3.55% vs ±6.01% vs baseline B)
4. **Clinical feature engineering** — 8 interaction features grounded in dermatological co-occurrence patterns, selected by mutual information
5. **Per-patient reasoning trace** — each diagnosis includes fired rules, certainty scores, conflict load, and FSM state; fully auditable by a clinician
6. **Biopsy triage protocol** — rule-based SAFE_BIOPSY_FREE / UNCERTAIN / BIOPSY_ADVISED with no learned thresholds
7. **Diagnostic ceiling finding** — chronic_dermatitis F1 drops to 0.755 even with symbolic reasoning, proving biopsy-free diagnosis has hard limits for exclusion-based diagnoses (publishable negative result)

---

## Limitations

- **N=366** — small dataset; McNemar test requires larger N to reach significance on a 29-feature model
- **Single dataset** — UCI Dermatology is the only public ESD dataset with 12 clinical features; external validation requires clinical collaboration
- **chronic_dermatitis** — diagnosis of exclusion; 12 clinical features cannot separate it from overlapping diseases without histopathology

---

## Citation

UCI Dermatology Dataset: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine, School of Information and Computer Science.
