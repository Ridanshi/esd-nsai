# HSCIS-ESD — Architecture Notes

Living document. Updated incrementally as we understand more of the pipeline.
Status: covers pipeline stages 1–6 in full (raw input → fuzzy grading → feature engineering → symbolic engine → CatBoost classifier → biopsy triage).
Not yet covered: evaluation layer (McNemar, ablation, SHAP), `app.py` as its own stage, Models A/B (out of scope by request — Model C is the project's actual contribution).

A single running patient example is used throughout Stages 4 and 6, so the numbers thread together:
```
erythema=0.667, scaling=0.667, definite_borders=0.333, itching=0.333,
koebner=1.0, knee_elbow=1.0, scalp=1.0, family_history=1.0,
polygonal=0.0, follicular=0.0, oral_mucosal=0.0, age=0.4375
```

---

## Full pipeline (updated)

```
[1] 12 raw clinical features (patient exam, no biopsy)                    ← COVERED
        │
        ▼
[2] FuzzyGrader — scales every raw value to [0,1]                          ← COVERED
        │
        ▼
[3] FeatureEngineer — derives 8 clinically-grounded interaction features    ← COVERED
        │
        ▼
[4] SymbolicPipeline                                                       ← COVERED
        ├── RuleEngine        — 45 rules → 6 per-disease certainty scores
        ├── ConflictAnalyzer  — conflict_load + contradiction_severity
        └── DiagnosticFSM     — fsm_state (0-4)
        │
        ├──────────────────────────────┬───────────────────────────────┐
        ▼ (all 29 features)            ▼ (symbolic outputs only, parallel — not sequential)
[5] CatBoost Classifier                [6] BiopsyTriage                  ← COVERED (both)
    10-fold CV, 200 oblivious trees        SAFE_BIOPSY_FREE / UNCERTAIN / BIOPSY_ADVISED
    predicted disease + confidence          (never sees CatBoost's output)
```

**Important structural correction from earlier drafts of this doc**: Stage 6 (`BiopsyTriage`) does **not** run after Stage 5 (CatBoost) using CatBoost's prediction. Verified directly in `app.py`'s `predict()` function — `BiopsyTriage.recommend()` is called with `top_certainty`, `conflict_load`, `fsm_state`, all pulled from the *symbolic engine's* output (`X_sym`), never from CatBoost's `predict_proba()`. The two run independently, side by side, off the same Stage 4 output.

---

## Stage 1 — Raw input: 12 clinical features

Source: `src/data/loader.py` — `CLINICAL_FEATURES` list. Pulled from UCI Dermatology dataset (id=33), 366 patients, restricted to the 12 features observable **without biopsy** (the other 22 of the full 34-feature dataset are histopathological — deliberately excluded, that exclusion is the whole premise of the project).

| # | Feature | Type | Scale | Clinical meaning |
|---|---|---|---|---|
| 1 | erythema | ordinal | 0–3 | skin redness / inflammation |
| 2 | scaling | ordinal | 0–3 | skin shedding / epidermal turnover |
| 3 | definite_borders | ordinal | 0–3 | sharp lesion edge vs. diffuse margin |
| 4 | itching | ordinal | 0–3 | pruritus intensity |
| 5 | koebner_phenomenon | binary | 0/1 | new lesions at trauma sites |
| 6 | polygonal_papules | binary | 0/1 | flat angular bumps — LP marker |
| 7 | follicular_papules | binary | 0/1 | perifollicular bumps — PRP marker |
| 8 | oral_mucosal_involvement | binary | 0/1 | mouth lesions — LP marker |
| 9 | knee_elbow_involvement | binary | 0/1 | extensor surface — psoriasis marker |
| 10 | scalp_involvement | binary | 0/1 | scalp lesions — seb derm / psoriasis |
| 11 | family_history | binary | 0/1 | HLA-Cw6-linked in psoriasis |
| 12 | age | continuous | 0–80 | disease incidence varies by age |

Missing `age` values median-imputed at load time (`loader.py`). No other missing-value handling needed — dataset is otherwise complete.

---

## Stage 2 — FuzzyGrader (`src/grading/fuzzy_grader.py`)

**What it is mechanically**: a hand-rolled min-max scaler with *fixed, clinically-defined* bounds — not sklearn's `MinMaxScaler` (which learns min/max from the training data), not `StandardScaler` (z-score would produce unbounded/negative values, incompatible with what comes next).

**Transform per feature type:**

| Type | Features | Formula |
|---|---|---|
| ordinal (0–3) | erythema, scaling, definite_borders, itching | `value / 3.0` |
| binary | all 7 binary features | unchanged (already 0/1) |
| continuous | age | `min(value / 80.0, 1.0)` |

Final `.clip(0.0, 1.0)` on the whole row as a safety net.

**Why "fuzzy" and not just "scaling"**: the scaling itself is mechanical. The *fuzzy logic* framing only becomes meaningful downstream, when these [0,1] values get treated as **degree of truth** (fuzzy set membership) rather than plain normalized numbers — specifically in the rule engine's `min()`-based AND logic (Stage 4). `FuzzyGrader` alone is just the scaler; it doesn't do anything fuzzy by itself.

**Output**: 12 fuzzy-graded features, all in [0,1], directly comparable to each other for the first time (a raw age of 45 and a raw erythema of 2 are now on the same footing: 0.5625 vs 0.667).

---

## Stage 3 — FeatureEngineer (`src/grading/feature_engineer.py`)

Takes the 12 fuzzy features and derives 8 additional interaction/composite features. Not learned — each one hand-designed from dermatology literature (Fitzpatrick's, Andrews'), then validated (not designed) by mutual-information scoring against the disease label (`select_features.py`, threshold MI≥0.05). One originally-designed feature (`young_adult`, age<35 flag) was dropped after scoring MI≈0 — kept in the "designed but data didn't support it" category.

| Feature | Formula | MI score | Captures |
|---|---|---|---|
| pso_triad | koebner × knee_elbow × family_history | 0.054 | psoriasis's near-unique 3-way combo (multiplication = fuzzy AND for binaries) |
| lp_classic | polygonal × oral_mucosal | 0.394 (highest) | lichen planus pathognomonic combo |
| prp_core | follicular × scaling | 0.202 | PRP signature |
| itch_no_border | itching × (1 − definite_borders) | 0.232 | chronic dermatitis pattern (itchy + fuzzy edges) |
| older_patient | 1.0 if age > 0.625 (50yr) else 0.0 | 0.081 | PRP adult-type / late-onset psoriasis |
| scale_erythema_ratio | scaling / (erythema + 0.01) | 0.144 | which sign dominates: psoriasis (scale-heavy) vs LP/chronic derm (erythema-heavy) |
| inflammation_burden | (erythema + scaling + itching) / 9 | 0.164 | overall severity, disease-agnostic |
| no_specific_morphology | 1 − clip(polygonal + follicular + koebner, 0, 1) | 0.345 | absence of specific signs → points toward seb derm / chronic derm by elimination |

**What "MI score" means:** Mutual Information — measures how much knowing a feature's value reduces uncertainty about the patient's actual disease. MI≈0 = feature is unrelated noise; higher MI = stronger relationship. Unlike Pearson correlation (which only catches linear relationships between two continuous variables), MI works for any relationship shape and handles a categorical multi-class label cleanly — the right tool here since the disease label has 6 unordered classes and some engineered features are step functions (e.g. `older_patient`), not smooth curves. Computed via `sklearn.feature_selection.mutual_info_classif(X_eng, y)` in `select_features.py`. Threshold for keeping a feature: MI ≥ 0.05 — this is how `young_adult` (age<35 flag) got cut, scoring MI≈0.0005.

**Two conceptually distinct patterns among these 8:**
- **AND-style combos** (`pso_triad`, `lp_classic`, `prp_core`, `itch_no_border`, `no_specific_morphology`) — multiplication/subtraction encoding "this specific combination of signs matters," same logic as the rule engine's threshold rules, just expressed as continuous formulas instead of discrete IF-THEN.
- **Ratio/composite features** (`scale_erythema_ratio`, `inflammation_burden`) — not AND logic, comparing or summing magnitudes rather than gating on presence/absence.

**Output**: 8 engineered features. Combined with Stage 2's 12 → 20 features so far, feeding into Stage 4.

---

## Stage 4a — RuleEngine (`src/symbolic/rule_engine.py`)

**Purpose**: apply 45 hand-written expert rules (`rules/*.yaml`, one file per disease) against the 12 fuzzy features only — **not** the 8 engineered features (verified: `grep -h "feature:" rules/*.yaml | sort -u` returns exactly the 12 raw clinical feature names, nothing from `FeatureEngineer`'s output). Produces one certainty score per disease — 6 numbers total, each in [0,1].

### Rule structure — 4 evidence tiers

| Tier | Baseline weight | Meaning |
|---|---|---|
| A | 1.0 | pathognomonic — near-certain for that disease |
| B | 0.6 | supportive — commonly associated |
| C | 0.3 | auxiliary — weakly associated |
| D | 0.4–0.5 | discriminating — sits in *this* disease's file but is actually a *competitor's* sign, subtracts from this disease's score when present |

Baseline weights are **not applied rigidly** — actual values deviate rule-by-rule based on hand judgment (e.g. `LIC_A03`, polygonal papules alone, gets 0.85 not the tier-A baseline 1.0). Confirmed via git history: no script in the repo ever calibrates rule weights against the dataset — pure textbook-sourced expert judgment (Fitzpatrick's, Andrews'), same tradition as classic expert systems like MYCIN. Flagged in `changes-made.md` §6 as a methodology gap, not a bug.

### How a single rule fires — two separate operations, using SEB_A02 as the example

```yaml
- id: SEB_A02
  conditions:
    - feature: scalp_involvement    threshold: 0.5
    - feature: itching              threshold: 0.33
    - feature: scaling              threshold: 0.33
  weight: 0.9
```
Patient values: `scalp=1.0, itching=0.333, scaling=0.667`

**Operation 1 — threshold gate** (pass/fail per condition, not a calculation):
```python
for cond in rule["conditions"]:
    if value < threshold:
        return 0.0     # any single failure kills the whole rule
    strengths.append(value)
```
All three conditions clear their bar → rule is allowed to fire.

**Operation 2 — strength** (only runs after all conditions passed):
```python
return float(np.min(strengths))    # = min(1.0, 0.333, 0.667) = 0.333
```
**Why `min`, not addition or average**: addition would let one strong sign (scalp=1.0) compensate for a missing one — wrong, because this rule represents a specific *triad* that must all be present together. `min` enforces "you're only as confident as your weakest required piece" — the standard fuzzy-logic AND operator (Zadeh's min-based T-norm).

Contribution to seb_derm's total: `0.333 × 0.9 (weight) = 0.300`.

### All 45 rules

Confirmed count: `grep -c "^- id:" rules/*.yaml` → psoriasis=8, seb_derm=7, LP=7, PR=7, chronic_derm=8, PRP=8 = **45 total**. (README/paper.md used to say 41 — documentation drift, `changes-made.md` §4, since corrected to 45 everywhere.)

**Psoriasis (8)** — PSO_A01 (koebner+knee+scalp, A/1.0), PSO_A02 (family+koebner+knee, A/1.0), PSO_B01 (scaling+erythema, B/0.6), PSO_B02 (family_history, B/0.6), PSO_B03 (family+scaling≥0.67, B/0.6), PSO_C01 (definite_borders, C/0.3), PSO_D01 (polygonal, D/0.4, vs LP), PSO_D02 (oral_mucosal, D/0.4, vs LP)

**Seborrheic dermatitis (7)** — SEB_A01 (scalp+scaling+erythema, A/1.0), SEB_A02 (scalp+itching+scaling, A/0.9), SEB_B01 (itching, B/0.6), SEB_B02 (scaling, B/0.6), SEB_C01 (definite_borders, C/0.3), SEB_D01 (koebner, D/0.4, vs psoriasis), SEB_D02 (knee_elbow, D/0.5, vs psoriasis)

**Lichen planus (7)** — LIC_A01 (polygonal+oral, A/1.0), LIC_A02 (polygonal+koebner, A/1.0), LIC_A03 (polygonal alone, A/0.85), LIC_B01 (itching, B/0.6), LIC_B02 (oral_mucosal, B/0.7), LIC_C01 (erythema, C/0.3), LIC_D01 (follicular, D/0.5, vs PRP)

**Pityriasis rosea (7)** — PIT_A01 (borders+scaling+erythema, A/1.0), PIT_B01 (itching, B/0.6), PIT_B02 (scaling, B/0.6), PIT_C01 (erythema, C/0.3), PIT_D01 (koebner, D/0.4, vs psoriasis), PIT_D02 (scalp, D/0.45, vs seb_derm), PIT_D03 (age≥50, D/0.35, vs psoriasis)

**Chronic dermatitis (8)** — CHR_B01 (itching+erythema, B/0.6), CHR_B02 (scaling, B/0.6), CHR_B03 (itching≥0.67+erythema+scaling, B/0.7), CHR_C01 (erythema≥0.5, C/0.3), CHR_D01 (koebner, D/0.4, vs psoriasis), CHR_D02 (polygonal, D/0.5, vs LP), CHR_D03 (oral_mucosal, D/0.5, vs LP), CHR_D04 (follicular, D/0.45, vs PRP)

**Pityriasis rubra pilaris (8)** — PRP_A01 (follicular+scaling, A/1.0), PRP_A02 (follicular alone, A/0.9), PRP_B01 (erythema+borders, B/0.6), PRP_B02 (scaling, B/0.6), PRP_C01 (itching, C/0.3), PRP_D01 (koebner, D/0.4, vs psoriasis), PRP_D02 (oral_mucosal, D/0.5, vs LP), PRP_D03 (polygonal, D/0.5, vs LP)

### Result for our running patient — all 6 certainty scores

Verified via `pipeline.explain()`:

| Disease | Certainty | Key driver |
|---|---|---|
| psoriasis | **0.7317** | PSO_A01 + PSO_A02 both fire at full strength (1.0 each), no D-tier penalties (no polygonal/oral present) |
| seborrheic_dermatitis | 0.2255 | 5 rules fire in support (sum 1.667), but SEB_D01+SEB_D02 penalize hard (−0.900) since patient has koebner+knee_elbow (psoriasis signs) |
| chronic_dermatitis | 0.0909 | some support, penalized by CHR_D01 (koebner) |
| pityriasis_rubra_pilaris | 0.0882 | mild support, penalized by PRP_D01 (koebner) |
| lichen_planus | 0.0449 | almost nothing fires — patient lacks polygonal/oral, LP's defining signs |
| pityriasis_rosea | 0.0000 | penalties (PIT_D01 koebner, PIT_D02 scalp) exceed support entirely — net negative, clipped to 0 |

---

## Stage 4b — ConflictAnalyzer (`src/symbolic/conflict.py`)

Takes the 6 certainty scores, produces two **separate, parallel** outputs — not sequential, both computed from the same input.

### `conflict_load` — "is this a hard case?"

**Purpose**: `max()` of the 6 scores tells you *who's leading*, not *how much to trust that lead*. Two patients can have the identical top score (0.75) — one a clean win, one a near-tie with a second disease — and `max()` can't tell them apart. `conflict_load` recovers that missing information.

**Mechanism, step by step:**
1. **Filter**: keep only diseases scoring `> 0.2` ("active" candidates) — for our patient, only psoriasis (0.7317) and seb_derm (0.2255) survive; everything else is too weak to count.
2. **Pair up survivors and multiply**: `psoriasis × seb_derm = 0.7317 × 0.2255 = 0.1650`
3. **Normalize by pair count**: `max_pairs = n(n-1)/2 = 1` → `conflict_load = 0.1650 / 1 = 0.165`

**Why multiply, not sum or min** (verified with 3 test cases): sum lets one big value alone inflate the score (fails to distinguish clean win from real conflict). `min` gets fooled by two diseases that are both merely weak (`min(0.21,0.21)=0.21`, wrongly reads as meaningful conflict). Multiplication is the only one that requires **both** values to be independently large — mirrors the probability formula for two independent events both being true: `P(A∩B) = P(A)×P(B)`.

**Why pairs, not an N-way product across all active diseases**: verified with a 3-disease case (0.50, 0.45, 0.40 all active) — a naive 3-way product gives 0.09 (misleadingly low), while pairwise-sum-then-average gives 0.2017 (correctly reflects real 3-way tension). Pairwise-and-average generalizes cleanly regardless of how many diseases are active; an N-way product doesn't (shrinks disproportionately as N grows, changes meaning depending on candidate count).

### `contradiction_severity` — "is this evidence internally inconsistent?"

**Purpose, distinct from conflict_load**: some disease pairs are not just "similar-looking" but **mutually exclusive by clinical definition** (`INCOMPATIBLE_PAIRS = [(psoriasis, lichen_planus), (pityriasis_rosea, pityriasis_rubra_pilaris)]`). If both members of one of these pairs score high simultaneously, that's not normal diagnostic ambiguity — it's a sign the evidence itself is telling two contradictory stories at once (e.g. strong psoriasis signs AND strong LP signs on the same patient — clinically shouldn't co-occur).

**Mechanism**: for each of the 2 fixed pairs, multiply the two scores; take the `max` across the (only 2) pair results — not sum. Verified with contrasting cases: one severe pair (0.90×0.85=0.765) vs two independent mild pairs (0.35×0.35=0.1225 each) — summing the two mild pairs would give 0.245, misleadingly implying it's worse than it is, since the two pairs share no diseases and are logically unrelated checks. `max` correctly reports "worst single violation found," appropriate because these are independent structural checks, not an accumulating pool of related competitors (unlike `conflict_load`'s active-disease set, which *is* one connected pool).

**For our patient**: `psoriasis × LP = 0.7317 × 0.0449 = 0.0329`; `PR × PRP = 0.0`. `contradiction_severity = max(0.0329, 0.0) = 0.0329` — negligible, no real contradiction.

### Important finding — `contradiction_severity` is more predictive than it looks, but underused

Empirically tested: removing `contradiction_severity` from Model C's 29-feature vector drops CV accuracy from 88.79%→87.72% and flips 10/366 predictions. Full CatBoost feature-importance ranking places it **#11 of 29** — ahead of both `fsm_state` (#13) and `conflict_load` (#16), the two symbolic signals that `BiopsyTriage` (Stage 6) actually uses. `BiopsyTriage` never receives `contradiction_severity` at all — confirmed via `grep`, and via git history (the value existed before `BiopsyTriage` was even written, so it wasn't a "didn't exist yet" gap). Logged as `changes-made.md` §7 — the most consequential finding so far, since it concerns the actual safety-critical decision layer. **Since fixed:** `contradiction_severity` is now a 4th input to `BiopsyTriage.recommend()`, gating `SAFE_BIOPSY_FREE`. The threshold value (`0.30`) remains a placeholder pending Ridanshi's clinical judgment.

---

## Stage 4c — DiagnosticFSM (`src/symbolic/fsm.py`)

**Purpose**: compress `top_certainty` + `conflict_load` into one ordered label (0-4) describing how far the reasoning has progressed — mimicking a doctor's confidence building through checkpoints, rather than reporting raw numbers. Used as: a hard gate in `BiopsyTriage` (requires `fsm_state == RESOLVED`), a clean single feature for CatBoost, and a plain-English summary in the app UI.

```python
class FSMState:
    EVIDENCE_SPARSE = 0; HYPOTHESIS_FORMING = 1; CERTAINTY_BUILDING = 2
    DIAGNOSTIC_TENSION = 3; RESOLVED = 4
```

**Gate 1** (`top_certainty > 0.1` → HYPOTHESIS_FORMING): 0.1 is the line between "nothing meaningful fired" and "at least one real signal exists" — below it typically means at most one weak C-tier rule barely cleared its threshold.

**Gate 2** (`top_certainty > 0.4` → CERTAINTY_BUILDING): 0.4 marks a genuine B-tier-strength signal, not just an isolated weak sign. Reused later as the "real contender" bar for disease-counting in Gate 3.

**Gate 3** (`conflict_load > 0.3` OR `2+ diseases scoring > 0.4` → DIAGNOSTIC_TENSION): two independent trip-wires, OR'd. Verified with a 3-disease-tied-at-0.42 case: `conflict_load` alone stays at 0.1764 (under the 0.3 bar, would miss it), but the disease-count check (3 ≥ 2) correctly catches it anyway. Two different failure patterns need two different checks.

**Gate 4 — the override** (`(top_certainty > 0.65 AND conflict_load < 0.25)` OR `top_certainty > 0.80` → RESOLVED): the **only** line in the function that ever assigns RESOLVED — there is no sequential gate 3→4. Checked unconditionally, regardless of current state; can even override a just-triggered DIAGNOSTIC_TENSION. Necessary because the sequential gates alone have no path from CERTAINTY_BUILDING to RESOLVED — without this override, a patient with strong, clean evidence would be stuck reporting "still building" forever. One sharp edge: branch 2 (`>0.80` alone) has no conflict requirement — an extremely confident patient can get forced to RESOLVED even with real unresolved tension still present.

**Full trace for our patient** (`top_certainty=0.7317`, `conflict_load=0.165`):
```
gate 1: 0.7317 > 0.1                              → HYPOTHESIS_FORMING (1)
gate 2: 0.7317 > 0.4                              → CERTAINTY_BUILDING (2)
gate 3: 0.165 > 0.3? no. 1 disease > 0.4? no       → stays (2)
gate 4: 0.7317>0.65 AND 0.165<0.25 → TRUE          → RESOLVED (4)
```

**Stage 4 output — all 9 symbolic features for our patient:**
```
certainty_psoriasis=0.7317, certainty_seb_derm=0.2255, certainty_LP=0.0449,
certainty_PR=0.0, certainty_chronic_derm=0.0909, certainty_PRP=0.0882,
conflict_load=0.165, contradiction_severity=0.0329, fsm_state=4
```
Combined with Stages 2-3's 20 features → **29 total features**, Stage 5's (CatBoost) input.

---

## Stage 5 — CatBoost Classifier (`src/models/model_c.py`, `src/models/base.py`)

**Purpose**: the rule engine already produces 6 interpretable certainty scores (Stage 4a) — so why do we need a statistical classifier on top? Because hand-written rules aren't calibrated to make the single best final call. CatBoost statistically learns, from 366 real patients, how to weigh and combine all 29 signals (12 fuzzy + 8 engineered + 9 symbolic) into the most accurate diagnosis — benefiting from the rules' structure without being limited to it. Concrete proof, our running patient: symbolic layer alone (`top_certainty`) gives 0.7317; CatBoost, using all 29 features, gives **0.9448**.

### What one tree actually is — verified against the real first tree

CatBoost uses **oblivious (symmetric) trees** — every patient is asked the *same* sequence of questions in the same order (unlike a standard tree, where different branches ask different questions). With `depth=3` (this project's value), every tree is exactly 3 fixed yes/no questions → 8 possible outcome buckets.

Real Tree 0, extracted from the trained model:
```
1. polygonal_papules ≥ 0.5 ?
2. scalp_involvement ≥ 0.5 ?
3. definite_borders  ≥ 0.5 ?
```
Our patient's real answers (NO, YES, NO) land in leaf bucket #2, which nudges the 6 disease scores by: `psoriasis +0.04, seb_derm +0.01, LP −0.02, PR −0.02, chronic_derm −0.02, PRP +0.01`. One tree's nudge is deliberately tiny — the whole point of boosting is stacking 200 of these.

### Random Forest vs. Gradient Boosting (CatBoost's family) — not the same kind of "many trees"

| | Random Forest | Gradient Boosting (CatBoost) |
|---|---|---|
| How trees are built | independently, in parallel | sequentially, each correcting what came before |
| What each tree targets | original labels, random data sample | the *error* still remaining after all previous trees |
| How trees combine | average / vote, equal weight | additive sum, each shrunk by `learning_rate` |
| Typical tree strength | deep, strong, low-bias | shallow, deliberately weak (`depth=3` here) |
| Reduces | variance (averaging cancels noise) | bias (iteratively whittles down error) |

### AdaBoost vs. CatBoost — different boosting mechanisms

AdaBoost (1997, the original) reweights **misclassified patients** after each round — the next tree is forced to pay more attention to them; trees are typically depth-1 stumps. CatBoost (like all modern gradient boosting) fits each new tree directly to the **residual/gradient of a loss function** — no sample reweighting, works with any differentiable loss, and can use deeper trees per-round (`depth=3` here vs. AdaBoost's depth=1). CatBoost adds two more things beyond generic gradient boosting: **ordered boosting** (reduces a subtle overfitting failure mode called prediction shift, especially relevant for this small 366-patient dataset) and the oblivious-tree structure already shown above. Note: CatBoost's namesake categorical-feature handling is *not* actually exercised in this project — all 29 features are continuous floats, not raw categoricals.

### Hyperparameters — `get_catboost_params_c()`, selected via a 108-combo sweep (`sweep_catboost.py`)

| Param | Value | Purpose |
|---|---|---|
| `depth` | 3 | tree size — matches the real 3-split tree shown above |
| `iterations` | 200 | how many trees get sequentially built |
| `learning_rate` | 0.05 | shrinks each tree's raw nudge before adding it to the running total |
| `l2_leaf_reg` | 5.0 | penalizes overconfident leaf values from small patient groups |
| `subsample` | 0.8 | each tree trains on a random 80% of patients, not all 366 |

These 5 were tuned specifically because this is a small dataset — `session.md` documents a pre-tuning train-val gap of **+12.5%** (real overfitting, observed and fixed once already). Everything else in the params dict (`random_seed`, `loss_function`, `eval_metric`, `bootstrap_type`, `verbose`) is operational config, not a tuned knob.

**Gap found**: `rsm` (feature-subsampling, CatBoost's equivalent of the old XGBoost config's `colsample_bytree=0.4`) sits unset at its default of 1.0 — silently dropped during the XGBoost→CatBoost migration. Along with two other untested knobs (`min_data_in_leaf`, `random_strength`), logged as `changes-made.md` §8.

**Also found**: `od_type="Iter"`/`od_wait=30` (the overfitting detector) is completely inert — `model.fit(X_train, y_train)` never passes an `eval_set`, so there's nothing for it to watch. Empirically verified: `od_wait=1` vs `od_wait=100000` both build all 200 trees, no difference. Logged as `changes-made.md` §9.

### Cross-validation — `StratifiedKFold`, why "stratified" isn't optional here

10-fold CV: split 366 patients into 10 groups, train on 9, test on the held-out 1, rotate — every patient tested exactly once, never on data it trained on. The reported `± 3.34%` is the standard deviation across those 10 individual fold accuracies (same "variance" concept the project's headline 45% variance-reduction claim is built on).

**Why stratified specifically matters, proven not asserted** — PRP has only 20 patients total. Ran both stratified and plain `KFold` on the real labels:
```
Plain KFold:   fold 4 → 0 PRP patients in validation (can't even measure PRP for that fold)
               fold 4's training set → all 20 PRP patients (extreme, opposite problem)
               fold 10 → 5 PRP patients in validation vs. the "fair" 2
StratifiedKFold: every single fold → exactly 2 PRP patients in validation, every time
```

### How predictions accumulate — real trajectory, not the expected shape

Traced our patient's psoriasis probability across the full 200-tree ensemble (`predict(..., ntree_end=N)` at checkpoints): **17.3% → 19.5% → 21.7% → 25.7% → 40.9% → 63.4% → 83.0% → 94.5%**. Not simple diminishing returns from tree 1 — it's an S-curve: slow start (near the 1/6≈16.7% "no information" baseline), fast middle (trees 5-100, real evidence compounding), slowing again near the end (100→200, softmax saturation near certainty).

### `_cross_validate_c()` — where the reported numbers actually come from

Trains **10 separate models**, one per fold (not one model reused) — plus an 11th, different model trained on all 366 patients with no held-out set at all for the actual `app.py` deployment. Within the loop: `accuracy_mean`/`std` and `macro_f1_mean`/`std` come from averaging the 10 individual fold scores (this is where "88.79% ± 3.34%" comes from, literally). But `per_class_f1` and the confusion matrix are computed differently — **pooled** from every patient's single prediction across all 10 folds combined, not averaged fold-by-fold.

**Verified this pooling choice is sound, not just described**: tested chronic_dermatitis (the project's hardest class) both ways — fold-level F1 swings from 0.600 to 0.923 across folds (std=0.089, real instability from small per-fold subgroups), yet mean-of-folds (0.7947) and pooled (0.7963) end up nearly identical either way. The real benefit of pooling isn't a different final number — it's that individual fold-level estimates are inherently noisy on their own (5-6 patients per fold), and a confusion matrix literally *cannot* be built by averaging 10 small per-fold matrices; pooling every patient's one true prediction is the only sound way to get one meaningful 6×6 grid. Pooled chronic_dermatitis F1 (0.7963) matches exactly what's published in `README.md`/`paper.md`, confirming this reproduces the project's real pipeline faithfully.

---

## Stage 6 — BiopsyTriage (`src/triage/biopsy_triage.py`)

**Purpose**: this is the actual deliverable of the whole project. Answers the one question everything else exists to support: *"can this patient be diagnosed without sending them for a biopsy?"* Outputs one of `SAFE_BIOPSY_FREE` / `UNCERTAIN` / `BIOPSY_ADVISED`.

**Runs in parallel to Stage 5, not after it** (see structural note at the top of this doc) — uses only `top_certainty`, `conflict_load`, `fsm_state` from Stage 4, never touches CatBoost's prediction.

**Fixed thresholds, not learned:**
```
SAFE_BIOPSY_FREE:  top_certainty ≥ 0.75 AND conflict_load < 0.20 AND fsm_state == RESOLVED
UNCERTAIN:         top_certainty ≥ 0.55 AND conflict_load < 0.40
BIOPSY_ADVISED:    otherwise
```

**For our patient**: `top_certainty=0.7317` — just short of the 0.75 SAFE bar, despite `fsm_state` already being RESOLVED and `conflict_load` being low. Falls to the UNCERTAIN check: `0.7317 ≥ 0.55` ✓ and `0.165 < 0.40` ✓ → **verified result: `UNCERTAIN`**. Demonstrates the threshold is deliberately conservative — "resolved" in the FSM's own judgment doesn't automatically mean "safe enough to skip biopsy" by the stricter triage bar.

**Known gap**: does not use `contradiction_severity` at all, despite that value being more predictive (by CatBoost's own feature-importance ranking) than the `fsm_state`/`conflict_load` it does use. Full detail in `changes-made.md` §7.

---

## Issues found during the audit (full detail + current status in `changes-made.md`)

> **Status note:** the list below is the *audit-phase snapshot*. Most of these have since been fixed on branch `hritwik/audit-fixes` — see `changes-made.md` for what was applied, what was tested-and-rejected, and the two items deliberately left as Ridanshi's decisions. The headline numbers (88.79% ± 3.34%) were verified unchanged by every applied fix.


1. `requirements.txt` missing `streamlit` + `catboost`
2. `session.md` stale — pre-CatBoost numbers
3. `inflammation_burden` divisor bug (`/9` should be `/3` for fuzzy inputs) — compresses feature to `[0, 0.333]`
4. Rule count documentation drift — docs say 41, actual is 45
5. `scale_erythema_ratio` unbounded (confirmed real max 33.33 on the dataset) — empirically zero accuracy impact currently, but flagged for interpretability/robustness reasons
6. Rule tier/weight values have no empirical calibration — methodology gap (textbook-sourced judgment only, unlike the MI-validated engineered features)
7. **`BiopsyTriage` ignores `contradiction_severity`** — despite it outranking the two signals the triage decision actually relies on. Most consequential finding — affects the safety-critical decision layer directly.
8. `rsm` (CatBoost's feature-subsampling regularization) silently dropped during the XGBoost→CatBoost migration — the old config's `colsample_bytree=0.4` never got carried over. Also flags `min_data_in_leaf` and `random_strength` as untested, theoretically-motivated knobs given PRP's 20-patient class.
9. Overfitting detector (`od_type`/`od_wait`) is configured but completely inert — no `eval_set` passed to `.fit()`, empirically verified to have zero effect regardless of setting.

*(Audit-phase statement, now superseded: at the time this list was written, none had been fixed. See the status note above.)*

Found later, during a second read-through, and **not** fixed — both are pre-existing errors in the docs themselves: Model B's published std and macro F1 don't reproduce (`changes-made.md` §11), and `paper.md`'s abstract claims per-class F1 improved on all six diseases when lichen planus is exactly tied (§12).

---

## Not yet covered in this document

- Models A and B — out of scope by explicit request; Model C is the project's actual contribution
- Why CatBoost specifically beat XGBoost for Model C — conceptual algorithm differences covered, but `catboost_trial.py`'s actual head-to-head comparison numbers not yet examined
- How raw scores become the final single prediction — softmax→argmax mechanics not explicitly named as their own step (seen implicitly via the probability trajectory)
- Evaluation layer: McNemar test, ablation study, SHAP/imodels explainability
- `app.py` — Streamlit inference UI (structure only referenced so far, not walked through as its own stage)
