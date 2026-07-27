# Changes Made

**Read this first if you are an agent or collaborator picking up this repo.**

## What this document is

This started as an audit log (`changes-tobe.md`) — 10 findings from a full read-through of the HSCIS-ESD pipeline, deliberately flagged and *not* fixed. It has since been converted into a record of what was actually implemented. Every finding below keeps its original evidence and reasoning, with a **STATUS** line stating what happened to it.

## Who did this, and what that means for you

**Ridanshi Agarwal (23BCI0026) wrote the entire HSCIS-ESD codebase.** Everything in `src/`, `rules/`, `app.py`, and the pipeline scripts is her work.

**Hritwik (23BAI0054) did the audit and the changes recorded here** — he did not write the original system. The audit phase was deliberately read-only; the implementation phase came later, only after explicit sign-off, and lives on a separate branch (`hritwik/audit-fixes`) so `main` stays untouched until Ridanshi reviews it.

**If you are a new agent working for Ridanshi:** do not assume the changes below are approved. They are *proposed and verified*, pending her review. Two items are explicitly left as her decisions (see "Still open" below) — do not silently resolve them.

## Status at a glance

| # | Finding | Status |
|---|---|---|
| 1 | `requirements.txt` missing `streamlit`/`catboost` | **DONE** — both added |
| 2 | `session.md` stale (pre-CatBoost) | **DONE** — §20/§21 appended |
| 3 | `inflammation_burden` divisor `/9`→`/3` | **DONE** — zero accuracy change, verified |
| 4 | Rule count drift (41→45) | **DONE** — README/paper/session + per-file table |
| 5 | `scale_erythema_ratio` unbounded | **DONE** — clipped at 3.0, zero accuracy change |
| 6 | Rule weights never empirically calibrated | **MEASURED ONLY** — `rules/*.yaml` untouched, Ridanshi's call |
| 7 | `BiopsyTriage` ignores `contradiction_severity` | **DONE** — wired in; threshold value still Ridanshi's call |
| 8 | `rsm`/`min_data_in_leaf`/`random_strength` untested | **TESTED, NO CHANGE** — current defaults already optimal |
| 9 | Overfitting detector inert | **DONE** — dead params removed (wiring `eval_set` was tested and *hurt*) |
| 10 | Explainability layer analyzed the wrong model | **DONE** — now trains real CatBoost |
| 11 | Model B's published numbers don't reproduce | **OPEN** — reported, not fixed |
| 12 | Abstract claims F1 improved on all six diseases | **OPEN** — reported, not fixed |
| 13 | Cosmetic leftovers (`catboost_trial.py`, `fuzzy_grader.py`) | **WON'T FIX** — deliberate, see #13 |

## Verification state

Re-run after every change, all green:
- `python -m pytest tests/ -q` → **48 passed** (was 43; 5 tests added, see #14)
- `python eval_run.py` → Model C **88.79% ± 3.34%**, macro F1 **0.8850**, McNemar p=0.0176

**The headline numbers did not move.** Findings #3, #5, #7 and #9 were all verified to leave CV accuracy bit-identical to the published 88.79%. Nothing in this document changes what the paper reports — except #11 and #12, which are pre-existing errors in the docs themselves, still unfixed.

## Still open — Ridanshi's decisions, do not auto-resolve

1. **The `0.30` contradiction threshold (#7).** A placeholder, not derived. There is no ground-truth label for "was SAFE_BIOPSY_FREE correct," so it cannot be optimized — it is a clinical risk-tolerance call. Currently inert on this dataset (no patient's triage label changes at `0.30`; closest sits at `0.272`).
2. **Whether to act on the rule-weight evidence (#6).** `ablation_rules.py` measures each rule's impact but changes nothing. Recalibrating 45 weights against 366 patients risks overfitting and weakens the paper's "expert-encoded, textbook-grounded" claim. See #6 for the full argument.
3. **The doc corrections (#11, #12).** Fixing #11 cascades into rewording the abstract's "44% variance reduction" headline to 43%. That is Ridanshi's paper to change.

## Things a new agent will otherwise waste time rediscovering

- **Local setup:** needs `pip3 install ucimlrepo catboost certifi`, and every dataset-fetching script must be prefixed with `SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")` on macOS python.org builds, or the UCI fetch fails on certs.
- **`catboost_info/`** regenerates on every `.fit()` call. Harmless, untracked, safe to `rm -rf`.
- **The ablation numbers are *supposed* to disagree with the headline.** `ablation.py` deliberately uses XGBoost (86.61%) as a fixed reference while deployed Model C is CatBoost (88.79%). This is documented in `paper.md` line 182 and in `base.py`'s `get_xgb_params_c()` docstring. It is not drift — do not "fix" it.
- **Tuning is exhausted.** Iterations, depth, learning rate, `rsm`, `min_data_in_leaf`, `random_strength`, k-fold count (5/10/15/20/30), RFECV feature pruning, and a symbolic→CatBoost cascade were all tested this session. Every one is neutral or worse than the current config. The train-val gap is +1.19%, i.e. the model is already at this dataset's ceiling. Don't re-litigate hyperparameters expecting gains.
- **There is no second dataset.** Verified, not assumed: every "other" ESD dataset found (Kaggle, ResearchGate, arff mirrors) is the same UCI 366 patients re-hosted — identical per-class counts. Image datasets (Fitzpatrick 17k, DDI) are the wrong shape for this tabular pipeline.
- **Companion docs:** `architecture.md` (full stage-by-stage technical reference, Stages 1–6) and `hritwik-session.md` (audit-phase handoff context).

---

## 1. `requirements.txt` missing `streamlit` and `catboost`

**File:** `requirements.txt`

**Problem:** Current code depends on both packages but neither is listed.
- `app.py` imports `streamlit`
- `src/models/base.py`, `src/models/model_c.py`, `catboost_trial.py`, `sweep_catboost.py`, `feature_prune.py` all import `catboost`

`requirements.txt` only lists `xgboost` (leftover from before the CatBoost switch).

**Why it matters:** `pip install -r requirements.txt` followed by `streamlit run app.py` or `python eval_run.py` will fail with `ModuleNotFoundError` on a clean environment. Anyone cloning the repo fresh (including Ridanshi on a new machine) hits this immediately.

**Fix — DONE:** Added `streamlit>=1.30.0` and `catboost>=1.2.0` to `requirements.txt`. `xgboost` deliberately kept — still used by Models A/B, `ablation.py`, `diagnose_fit.py`, `sweep_params.py`.

---

## 2. `session.md` is stale — shows pre-CatBoost numbers

**File:** `session.md`

**Problem:** Documents Model C at 86.61% ± 3.55% (XGBoost, tuned 2026-07-11). Actual current model (per `README.md`, `paper.md`, `app.py`) is CatBoost at 88.79% ± 3.34%, switched in commit `c304290` and tuned further in `1c56f4f`. `session.md` also doesn't mention `app.py` (Streamlit UI, commits `937c6c8` onward) or the reasoning-trace UI iterations (5 commits polishing it).

**Why it matters:** `session.md` is the dev log — if Ridanshi or a collaborator reads it to catch up on state, it describes a model and feature set that's one full architecture generation behind what's actually in the repo.

**Fix — DONE:** Added §20 (CatBoost migration + tuning + Streamlit app, with commit hashes and deltas) and §21 (re-checked "What Is Left") to `session.md`, and corrected its header date. Original intent below.

**Fix:** Append a "2026-07-1X — CatBoost switch + Streamlit app" section to `session.md` covering: XGBoost→CatBoost swap and why (accuracy delta, gap delta from `catboost_trial.py`), final tuned CatBoost params (from `sweep_catboost.py`), `app.py` addition, and the "What Is Left" section should be re-checked against current state.

---

## 3. `inflammation_burden` divisor bug

**File:** `src/grading/feature_engineer.py`, line 53

```python
"inflammation_burden": (erythema + scaling + itching) / 9.0,
```

**Problem:** By the time this runs, `erythema`, `scaling`, `itching` are already fuzzy-graded to `[0,1]` (this function only ever receives `FuzzyGrader` output — confirmed via `model_c.py`: `FeatureEngineer().engineer(X_fuzzy)` runs strictly after `grader.grade(X_clinical)`). Max possible sum of three `[0,1]` values is **3**, not 9. The `/9.0` divisor caps this feature at **0.333** instead of the intended `[0,1]` range — it can never signal "maximum inflammation."

Git history confirms this was never intentional: the divisor has been `/9.0` unchanged since the very first commit (`a131d09`), when the comment simply said "across the three cardinal symptoms." A later commit (`de22fde`) updated the comment to `/ max`, implying whoever touched it believed 9 *was* the correct max — it isn't, for fuzzy-graded input. Most likely leftover from an earlier design where this ran on raw ordinal 0–3 scores (3 features × max 3 = 9), before the pipeline order settled on fuzzy-first.

**Why it matters:** Doesn't break tree-model training (CatBoost/XGBoost split on relative thresholds, so ordering between patients is preserved regardless of absolute scale) — MI score of 0.164 was measured on this buggy version and still cleared the keep threshold. But: the feature's stated semantics ("burden," implying it can reach full severity) don't match its actual behavior, and any future consumer expecting `[0,1]` scale (a non-tree model, a distance-based method, manual interpretation of raw feature values in `trace.py` output) will silently get wrong numbers.

**Fix — DONE:** Divisor changed `9.0` → `3.0`. Retrained and re-ran `eval_run.py`: **88.79% ± 3.34%, macro F1 0.8850 — bit-identical to before.** The caveat below was checked and came back clean, exactly as the reasoning predicted (tree models split on relative thresholds, so rescaling one feature monotonically changes nothing). Now covered by a regression test (see #14).

**Caveat before fixing:** Changing this will shift `inflammation_burden`'s actual values fed to CatBoost, meaning current reported CV accuracy (88.79%) is not guaranteed to hold — retraining and re-running `eval_run.py` after the fix is required to confirm no regression, and README/paper.md numbers may need updating if it changes.

---

## 4. Rule count documentation drift — README says 41, actual is 45

**Files:** `README.md`, `paper.md`, `session.md` (all say "41 rules") vs `rules/*.yaml` (actual count)

**Problem:** Counting rules directly in each YAML file:
- psoriasis.yaml: 8 (not 7 as README's per-file table claims)
- seborrheic_dermatitis.yaml: 7
- lichen_planus.yaml: 7
- pityriasis_rosea.yaml: 7
- chronic_dermatitis.yaml: 8
- pityriasis_rubra_pilaris.yaml: 8

Total = 45, not 41.

**Why it matters:** Minor, but this number appears in `paper.md` (the actual submission draft) multiple times as a stated fact ("41 expert-encoded rules"). If submitted as-is, a reviewer who counts rules in the released rule files will find a mismatch between the paper's claimed count and the artifact.

**Fix — DONE:** All "41" references updated to "45" across `README.md`, `paper.md`, `session.md`, and README's per-file table corrected (psoriasis.yaml 7 → 8). Verified zero remaining "41 rule"/"41 expert" occurrences.

---

## 5. `scale_erythema_ratio` unbounded — real outliers confirmed on the actual dataset

**File:** `src/grading/feature_engineer.py`, line 51

```python
"scale_erythema_ratio": scaling / (erythema + 0.01),
```

**Problem:** Every other engineered feature in this file is bounded to `[0,1]` — either naturally (products of `[0,1]` values) or by explicit `np.clip(...)` (`no_specific_morphology`). This one is a bare ratio with no clip, so it's unbounded whenever `erythema` is low.

**Verified against the real 366-patient UCI dataset** (not just theoretical domain enumeration):

| Metric | Value |
|---|---|
| Patients with erythema=0 | 4 / 366 |
| Max `scale_erythema_ratio` observed | **33.33** |
| Mean `scale_erythema_ratio` | **1.1566** (already above 1.0 — every other engineered feature's mean is under 0.5) |
| Patients with ratio > 1.0 | 45 / 366 (12.3%) |
| Patients with ratio > 5.0 | 3 / 366 |

Full range check across all 8 engineered features on real data, for comparison:

```
pso_triad                 min=0.0000  max=1.0000
lp_classic                min=0.0000  max=1.0000
prp_core                  min=0.0000  max=0.6667
itch_no_border            min=0.0000  max=1.0000
older_patient              min=0.0000  max=1.0000
scale_erythema_ratio       min=0.0000  max=33.3333   ← outlier
inflammation_burden        min=0.0000  max=0.3333    ← bug #3 above
no_specific_morphology     min=0.0000  max=1.0000
```

**Empirically tested against one specific model — result was zero effect on that run, but this does not clear the finding.** Ran the actual Model C CatBoost pipeline (`get_catboost_params_c()`, identical 10-fold CV split, `RANDOM_STATE=42`) twice: current unbounded ratio vs. clipped to 3.0. Result: identical accuracy (88.79%), identical macro F1, identical predictions on all 366 patients. Mechanistic reason: tree ensembles split on a feature's *relative order* within its own column, not cross-feature magnitude, so this specific outlier happened not to matter for this specific trained configuration.

**Why that single result doesn't make this low-priority:**
- It only tests *today's* fixed hyperparameters and *today's* fixed CV split. This repo actively re-sweeps hyperparameters (`sweep_catboost.py`, `sweep_params.py`, `catboost_trial.py` all exist and get re-run) — a different `depth`, `iterations`, or random seed can put split boundaries in different places, and there is no guarantee the outlier stays inert under a different configuration next time someone tunes.
- **Interpretability is a stated contribution of this project, not a side feature.** `explainability.py` computes mean `|SHAP value|` across all 29 features specifically to produce the feature-importance ranking the paper cites as evidence of "auditable reasoning" (`paper.md` §4.3, §6.1). SHAP magnitude is sensitive to a feature's raw value range in ways split *selection* isn't — this was not tested here, and given the project's explainability claims rest partly on this exact computation, an unverified distortion risk here is not a minor concern.
- **Production robustness is untested and real.** `app.py` is a live inference surface, not just an evaluation script. Training data tops out at ratio=33.33 (3-4 patients); a real incoming patient with erythema=0 and high scaling produces ratio=100 — outside anything the model has ever partitioned on. The model won't crash, but it also can't meaningfully discriminate that patient from the training-set outliers it's never actually seen this value from. That's a live-system correctness gap, not a benchmark footnote.
- Every other engineered feature in this file respects a `[0,1]` convention, either naturally or via explicit `clip()`. This is the one exception, and it's an exception nobody consciously chose (established via git history — no commit ever discusses this feature's range). An unintentional deviation from a project-wide convention is worth fixing on consistency grounds alone, independent of whether one CV run happened to absorb it.

**Fix — DONE:** Clipped to `min(scaling / (erythema + 0.01), 3.0)`. Retrained: **88.79% ± 3.34% — unchanged**, confirming the empirical prediction below. Now covered by a regression test (see #14). Original reasoning retained:

**Fix:** Clip the ratio, e.g. `min(scaling / (erythema + 0.01), 3.0)`, matching the `[0,1]`-family convention every other feature in this file follows. Recommended, not optional — the empirical test rules out "this broke the current reported number," it does not rule out the interpretability and production-robustness concerns above.

---

## 6. Rule tier/weight values have no empirical calibration — methodology gap, not a bug

**Files:** `rules/*.yaml` (all 45 rules)

**Observation:** Every rule's `tier` (A/B/C/D) and `weight` value is assigned by hand from clinical literature (Fitzpatrick's Dermatology, Andrews' Diseases of the Skin, cited at the top of each YAML file) — confirmed by exhaustive search: no script in this repo (`grep -rln "weight" --include="*.py"` across the whole project) ever computes, sweeps, or validates a rule weight against the actual 366-patient dataset.

**The convention isn't even applied consistently as a fixed scale.** README documents a clean tier→weight mapping (A=1.0, B=0.6, C=0.3, D=0.4), but actual values deviate rule-by-rule:
- Tier A weights observed: 1.0, 1.0, 0.85, 0.9, 1.0, 1.0, 0.9, 1.0, 1.0
- Tier B weights observed: 0.6, 0.6, 0.6, 0.7, 0.6, 0.6, 0.6, 0.7, 0.6, 0.6
- Tier D weights observed: 0.4, 0.4, 0.4, 0.5, 0.45, 0.4, 0.35, 0.4, 0.5, 0.5, 0.45, 0.4, 0.5, 0.5

Confirmed via git history this is intentional hand-tuning, not drift: commit `b6e6516` shows `LIC_B02` explicitly revised from `weight: 0.6` to `weight: 0.7` with the added comment *"Oral mucosal involvement (Wickham's striae) is highly specific to lichen planus"* — a judgment call, not a computed adjustment.

**Why this is worth flagging (as methodology, not a defect):** the rest of this pipeline holds itself to an empirical-validation standard — the 8 engineered features were hand-designed *then* checked against the dataset via mutual information scoring (`select_features.py`, MI≥0.05 threshold, one feature dropped for failing it); CatBoost hyperparameters went through a 108-combination CV sweep (`sweep_catboost.py`). Rule weights are the one component of the 29-feature pipeline that skips this step entirely — designed from textbook authority and never checked against outcomes (e.g. "does bumping LIC_B02 to 0.7 actually improve lichen planus classification, or reduce error elsewhere?" is never asked). This is a legitimate, traditional way to build expert systems (MYCIN, cited in `paper.md`'s own related work, worked the same way) — not inherently wrong — but it is an asymmetry in rigor between two parts of the same system that's worth Ridanshi being aware of, particularly since `paper.md` markets the rule library as "expert-encoded" without noting this contrast to the empirically-validated feature engineering step sitting right next to it.

**Not proposing a fix here** — recalibrating rule weights against outcome data would be a nontrivial methodology change and is a judgment call for Ridanshi to decide is worth doing, not something to silently change.

**Measurement tool built and run — `ablation_rules.py`, no weights changed.** Zeroes each of the 45 rules' weight one at a time (in memory only, `rules/*.yaml` untouched), rebuilds the full 29-feature pipeline, reruns identical 10-fold CV CatBoost, restores the weight, moves to the next rule. Baseline: acc=0.8879, macroF1=0.8850.

Top 10 by accuracy impact when individually removed:

```
Rule       Disease                    Tier  Weight    AccΔ      F1Δ
PSO_D01    psoriasis                     D    0.40  +0.0191  +0.0200
SEB_B02    seborrheic_dermatitis         B    0.60  +0.0137  +0.0155
PSO_B01    psoriasis                     B    0.60  +0.0136  +0.0126
PSO_D02    psoriasis                     D    0.40  +0.0136  +0.0151
PRP_B02    pityriasis_rubra_pilaris      B    0.60  +0.0136  +0.0196
PSO_A01    psoriasis                     A    1.00  +0.0135  +0.0171
PIT_A01    pityriasis_rosea              A    1.00  +0.0110  +0.0126
PRP_D02    pityriasis_rubra_pilaris      D    0.50  +0.0109  +0.0127
PIT_D01    pityriasis_rosea              D    0.40  +0.0108  +0.0115
LIC_B01    lichen_planus                 B    0.60  +0.0083  +0.0080
```

**Findings:**
- **30/45 rules show measurable impact (|Δacc| > 0.5pp) when individually ablated; 15 show none.** No rule is net-harmful — nothing crosses below −0.5pp (closest: `CHR_B01` at −0.28pp, `SEB_D02` at −0.27pp, both below the noise threshold). No rule is an obvious candidate for removal on accuracy grounds.
- **The impact ranking doesn't track the tier convention.** The single highest-impact rule is `PSO_D01` — a **D-tier** rule at baseline weight 0.40 — outranking every A-tier pathognomonic rule including `PSO_A01` (weight 1.00, ranks #6). Tier is meant to encode "how strongly this rule should count" (A=near-certain, D=discriminating/lower), but empirical CV impact doesn't track that ordering cleanly. This is direct evidence for the exact gap this finding describes: tier/weight assignment is textbook-sourced judgment, not outcome-validated, and here's a concrete case where the two diverge.
- This is evidence for Ridanshi's review, not a conclusion to act on unilaterally — ablating one rule at a time doesn't capture interaction effects between rules, and "impactful when removed" isn't the same claim as "current weight value is wrong." Full output: `ablation_rules.py`.

---

## 7. `BiopsyTriage` ignores `contradiction_severity` — despite it being more predictive than the two signals it does use

**File:** `src/triage/biopsy_triage.py`

**Observation:** `BiopsyTriage.recommend()` decides the entire clinical output of this project — `SAFE_BIOPSY_FREE` / `UNCERTAIN` / `BIOPSY_ADVISED` — using exactly 3 inputs: `top_certainty`, `conflict_load`, `fsm_state`. It never receives or checks `contradiction_severity`, even though that value exists specifically to flag "the evidence is claiming two clinically-impossible diseases at once" — arguably the single most relevant signal for a triage decision about whether to trust the model or send the patient for biopsy.

**Confirmed not a deliberate exclusion — a gap:**
- `git log --follow -- src/triage/biopsy_triage.py` shows exactly **one commit in the file's entire history** (`2e23294`, initial creation). Never revised since.
- Chronological check: `ConflictAnalyzer` (commit `4c9d884`, which computes `contradiction_severity`) was written **before** `BiopsyTriage` (`2e23294`). The value was already fully computed and available at the moment `BiopsyTriage` was designed — it wasn't a case of the signal not existing yet.
- No commit message, code comment, `session.md` entry, or paper section ever discusses why it was left out.
- `session.md`'s "What Is Left" list has a related open item — *"Threshold sweep for BiopsyTriage — vary SAFE threshold 0.60–0.85"* — showing Ridanshi was thinking about tuning the *existing 3 inputs'* threshold values, never about adding a 4th input.

**Empirically verified this isn't a trivial signal to leave out.** Trained the real Model C CatBoost pipeline twice — with and without `contradiction_severity` in the 29-feature vector (identical CV split, identical hyperparameters):

| | Accuracy | Macro F1 |
|---|---|---|
| WITH `contradiction_severity` | **88.79% ± 3.34%** | **0.8850** |
| WITHOUT `contradiction_severity` | 87.72% ± 3.65% | 0.8664 |

Removing it costs ~1.07pp accuracy and flips predictions on 10/366 patients (7 net worse, 3 net better). Full CatBoost feature-importance ranking (all 29 features, trained on the full dataset):

```
 1. knee_elbow_involvement            13.28
 ...
 9. certainty_chronic_dermatitis       4.38
10. definite_borders                   3.41
11. contradiction_severity             3.10   ← ranks here
12. oral_mucosal_involvement           2.59
13. fsm_state                          2.58   ← BiopsyTriage DOES use this
    ...
16. conflict_load                      1.95   ← BiopsyTriage DOES use this
    ...
29. pso_triad                          0.00
```

**`contradiction_severity` ranks #11 of 29 — outranking both `fsm_state` (#13) and `conflict_load` (#16), the two symbolic signals `BiopsyTriage` actually relies on.** The safety-critical decision in this entire project is built on two inputs that are demonstrably less informative (by CatBoost's own measure) than a third input sitting unused, right next to them, in the same 9-output symbolic pipeline.

**Why this matters more than the other findings above:** this isn't a cosmetic or interpretability concern — `BiopsyTriage` is the actual deliverable of the project (the "biopsy-free" decision). A patient whose evidence contains a genuine internal contradiction (e.g. our earlier Case X: psoriasis=0.90 and lichen_planus=0.85 simultaneously, clinically impossible together) could still receive a confident, low-conflict `SAFE_BIOPSY_FREE` verdict if `conflict_load` and `fsm_state` alone don't happen to catch it — the one signal purpose-built to catch exactly that scenario is never consulted.

**Fix — implemented:** `contradiction_severity` added as a 4th parameter to `BiopsyTriage.recommend()` and `batch_recommend()` (`src/triage/biopsy_triage.py`), gating `SAFE_BIOPSY_FREE` on `contradiction_severity < SAFE_CONTRADICTION_THRESHOLD` (set to `0.30`, mirroring the existing `conflict_load < 0.20` pattern). `app.py`'s `predict()` updated to pass it through and display it in the UI caption alongside conflict load.

**The `0.30` value is a placeholder, not a derived number** — same status as the example in this finding originally. Ridanshi should review and pick the real threshold deliberately; this just makes the parameter exist and wires it through everywhere it needs to flow.

**Empirically checked impact on the real 366-patient dataset:** zero patients' triage label actually changes at `0.30` — none of the 17 currently-`SAFE_BIOPSY_FREE` patients cross it (closest: `0.272`, margin of `0.028` below threshold). Dataset-wide max `contradiction_severity` is `0.586`, and 6/366 patients exceed `0.30` overall, but none of those 6 also satisfy the other SAFE conditions (certainty≥0.75, conflict<0.20, fsm=RESOLVED) simultaneously — so on this specific dataset, the gate is a currently-inert safety net, not a currently-active one. Same shape as findings #3/#5: doesn't change today's numbers, but closes a real gap for patients this exact dataset doesn't happen to contain (one SAFE patient sits only 0.028 below the line — plausible for a similar future patient to cross it).

---

## 8. CatBoost's feature-subsampling regularization was silently dropped during the XGBoost→CatBoost migration

**File:** `src/models/base.py` — `get_catboost_params_c()`; also `sweep_catboost.py`

**Observation:** the pre-CatBoost XGBoost config (`get_xgb_params_c()`, same file) used `colsample_bytree=0.4` — a deliberate overfitting guard that trains each tree on a random 40% of the 29 features rather than all of them. CatBoost's direct equivalent is `rsm` (Random Subspace Method). Verified via `CatBoostClassifier().get_all_params()`: `rsm` is not set anywhere in `get_catboost_params_c()` or in `sweep_catboost.py`'s grid, so it silently sits at CatBoost's **default of 1.0** — every tree uses all 29 features, every time. This specific regularization technique existed in the model before the swap and was never carried over, deliberately or otherwise — no commit message or doc mentions dropping it.

**Broader gap, same root cause:** `sweep_catboost.py`'s grid only tunes 5 of the ~35 parameters CatBoost exposes (`iterations`, `depth`, `l2_leaf_reg`, `learning_rate`, `subsample`). Two other untested knobs stand out as specifically relevant to this dataset's shape (366 patients total, `pityriasis_rubra_pilaris` has only 20):
- **`min_data_in_leaf`** (default 1) — a leaf can currently form around a single patient, a direct overfitting signature, most risky for the smallest class
- **`random_strength`** (default 1) — adds randomness to split-scoring, a cheap regularization axis orthogonal to the 5 already swept

**Explicitly not recommending re-adding**: `auto_class_weights` — already tested (on the pre-CatBoost XGBoost model, documented in `session.md`) and made results worse (85.52%→84.70%), because chronic_dermatitis's weakness is genuine feature overlap between diseases, not class-count imbalance. That diagnosis is about the data, not the algorithm, so it would likely still hold on CatBoost — not worth re-testing without new reason to doubt it.

**Why this matters:** unlike findings #3/#5 (engineered-feature bugs, empirically shown to not affect current accuracy), this one is specifically about the model's *overfitting defenses* on a 366-patient dataset — the exact concern the whole 108-combo sweep was built to address (`session.md` documents a pre-tuning train-val gap of +12.5%, i.e. real overfitting was observed and fixed once already). Losing `rsm`'s feature-subsampling during the CatBoost migration is a regression in that same defense, not a new problem — it re-opens a door the original XGBoost tuning had closed.

**Run — `sweep_catboost_expanded.py`, 36 combos × 10-fold CV, holding the existing best 5 params fixed:**

```
Current baseline (rsm=1.0, mdl=1, rs=1): ValAcc=0.8879  F1=0.8850  Gap=+0.0119
Best found:                              ValAcc=0.8879  F1=0.8850  Gap=+0.0119  (rsm=1.0, mdl=1, rs=1)
Delta vs baseline: +0.0000 (+0.00pp)
```

**Result: no gain, current defaults already win.** Every tested `rsm<1.0` value strictly hurts (0.8825 at rsm=0.4, down to lower at 0.6/0.8); every tested `random_strength>1` hurts (0.8797 at rs=2); `min_data_in_leaf` has **zero measurable effect** at any tested value (1/3/5 — identical results in every row) because at `depth=3`, trees never form leaves small enough for the knob to bind. No code change made — `get_catboost_params_c()` stays as-is, this closes the finding rather than opening a fix. Consistent with #5 and the RFECV feature-pruning test run this session: this project's hyperparameter and feature space is already fully exploited, there's no accuracy left to reclaim from tuning.

---

## 9. Overfitting detector (`od_type`/`od_wait`) is configured but completely inert — no `eval_set` for it to watch

**File:** `src/models/model_c.py`, line 46; hyperparameters defined in `src/models/base.py` — `get_catboost_params_c()`

**Observation:** `get_catboost_params_c()` sets `od_type="Iter"` and `od_wait=30` — CatBoost's early-stopping mechanism, meant to halt training if a validation metric hasn't improved for 30 consecutive tree-additions. But `model_c.py`'s actual training call is `model.fit(X_train, y_train)` — **no `eval_set` is passed.** CatBoost's overfitting detector requires a held-out validation slice to monitor during training; without one, there is nothing for it to watch.

**Empirically verified, not just inferred from the missing argument:** trained the same model with `od_wait=1` (should stop almost immediately if it ever triggers) vs `od_wait=100000` (effectively disabled) — both produced **`model.tree_count_ == 200`**, i.e. all 200 trees always get built regardless of the setting. The parameter has zero observable effect in this codebase.

**Why this matters:** this is the same shape of finding as `contradiction_severity` (#7) and `rsm` (#8) — a mechanism that exists, is configured with a specific deliberate-looking value (30, not a default), and does nothing. Unlike those two, this one isn't a missed opportunity to use an already-computed signal — it's a safety mechanism that *looks* present in the hyperparameter dict (and could mislead someone reading `get_catboost_params_c()` into thinking overfitting protection is active during training) but provides no actual protection. The real overfitting defense in this project is entirely the *fixed* hyperparameter values themselves (shallow `depth=3`, high `l2_leaf_reg=5.0`, etc., found via the 108-combo sweep) — not this detector, despite its presence suggesting otherwise.

**Fix — DONE, via the second option, after the first was tested and rejected.**

Both options from the original recommendation were tried:

*Option A — wire up a real `eval_set` (REJECTED, empirically).* Carved a stratified 15% validation slice out of each fold's training portion. Result: **accuracy collapsed 88.79% → 81.69%**, PRP F1 fell 0.9756 → 0.7097, and Model C lost statistical significance vs Model B entirely (McNemar p=0.0176 → p=0.1272). Isolated the cause with a three-way test:

```
baseline (no eval_set, od present but inert)  acc=0.8879  trees=[200 x10]
eval_set carved, od REMOVED (no early stop)   acc=0.8635  trees=[127,107,67,79,52,129,85,141,124,127]
eval_set carved, od ACTIVE                    acc=0.8169  trees=[28,107,23,44,52,44,32,23,40,7]
```

Note the middle row: passing an `eval_set` *at all* costs accuracy even with the detector disabled, because CatBoost's implicit best-model truncation kicks in. On 366 patients (PRP=20) there is simply no data to spare for a validation slice, and the train-val gap was only +1.19% — there was no overfitting to reclaim. Making the detector functional would have solved a problem that doesn't exist here at the cost of one that does. All changes reverted.

*Option B — remove the dead params (APPLIED).* `od_type`/`od_wait` deleted from `get_catboost_params_c()` in `src/models/base.py`, with the reasoning recorded in its docstring so nobody re-adds them. Also stripped from `sweep_catboost.py`'s inline config. Verified: **88.79% ± 3.34% restored exactly**, McNemar p=0.0176.

Left alone deliberately: `catboost_trial.py` still hardcodes `od_type`/`od_wait` (see #13).

**Fix (original recommendation):** either wire up a real `eval_set` (e.g. carve out a validation split inside each CV fold's training portion) so the detector can actually function, or remove `od_type`/`od_wait` from the params entirely to avoid the misleading impression that early-stopping protection is active. Your friend should decide which — adding a real eval_set changes what each fold's model actually sees during training (slightly less data to fit on), which could shift results from the currently-published numbers.

---

## 10. Explainability layer (SHAP + `imodels` rules) never updated after the CatBoost migration — analyzes the wrong model

**Files:** `src/evaluation/explainability.py`, `notebooks/analysis.ipynb`

**Observation:** `explainability.py` (`train_final_model`, `compute_shap_values`, `plot_shap_global`, `plot_shap_beeswarm`, `extract_imodels_rules`) is the code that generates this project's SHAP importance plots and `imodels` RuleFit extraction — the interpretability evidence `paper.md` cites (§4.3, §6.1). It imports and trains `XGBClassifier` (`train_final_model`, line 13), using `get_xgb_params()` (`base.py:17`) — not `get_catboost_params_c()`. Model C has been CatBoost since commit `c304290`. Every SHAP chart and extracted rule this code produces describes an XGBoost model, not the deployed one.

**Worse than just the wrong algorithm:** `get_xgb_params()` is the *generic* XGBoost config (`max_depth=4`, `colsample_bytree=0.8`) — not even `get_xgb_params_c()`, the XGBoost-era tuned config for Model C (kept deliberately, see §8/§ablation note below). So this was never analyzing Model C's real hyperparameters either, XGBoost or CatBoost, at any point in the project's history.

**Confirmed real leftover, not a deliberate choice** — checked via git dates:
- `explainability.py` and `notebooks/analysis.ipynb` last touched: commits `db68aee`/`3882352`/`31b88a6`, **2026-07-12**
- CatBoost migration: `c304290`, **2026-07-15 20:00**; final tuning: `1c56f4f`, **2026-07-15 21:01**
- Neither file touched since. No commit message, comment, or doc discusses it.

This is a genuinely different situation from `ablation.py`'s XGBoost use, which is **intentional and documented** — migration commit `c304290`'s message says "keep `get_xgb_params_c()` for ablation reference," and `base.py:30`'s docstring says the same ("kept for reference/ablation"). Only `explainability.py`/the notebook are the unaddressed gap.

**Compounds the confusion**: `notebooks/analysis.ipynb` line 187 assigns the result to a variable named `final_model_c` (`final_model_c = train_final_model(res_c["X_combined"], y)`) — the naming implies it's the real Model C, but `train_final_model` trains plain XGBoost underneath.

**Not a contamination risk for finding #7**: the `contradiction_severity` #11/29 importance ranking used elsewhere in this doc was computed via a separate ad hoc script against the real CatBoost model this session (raw importance values ~13.28, 4.38, etc. — CatBoost's native `PredictionValuesChange` scale, not SHAP magnitude) — not via this stale file.

**Why this matters:** interpretability is a stated contribution of the project (`paper.md` §4.3, §6.1 cite SHAP-based feature importance as evidence of "auditable reasoning"). If any currently-saved SHAP plots or extracted rule lists came from this code, they describe a model that was never deployed — a direct mismatch between what the paper claims to explain and what the explainability code actually explains.

**Fix — DONE:** `train_final_model()` in `src/evaluation/explainability.py` now builds `CatBoostClassifier(**get_catboost_params_c())` instead of `XGBClassifier(**get_xgb_params())`. Verified end-to-end that SHAP still works against CatBoost: `TreeExplainer` returns valid values, `list` of 6 classes, shape `(366, 29)` each. `notebooks/analysis.ipynb` needed no edit — it already calls `train_final_model()` and names the result `final_model_c`, which is now accurate; re-running the notebook regenerates correct plots automatically.

**Not yet done:** the notebook has not been re-executed top-to-bottom, so any *saved* SHAP images still on disk are stale until someone reruns it.

**Fix (original recommendation):** update `train_final_model` to train `CatBoostClassifier(**get_catboost_params_c())` instead of `XGBClassifier`. SHAP's `TreeExplainer` supports CatBoost natively, so the rest of `compute_shap_values`/`plot_shap_global`/`plot_shap_beeswarm` should work with minimal change — `imodels`' `RuleFitClassifier` is algorithm-agnostic (fits its own model), so `extract_imodels_rules` is unaffected either way. Any existing saved SHAP plots (`shap_global.png`, `shap_beeswarm.png`, etc.) or rule lists in the notebook output should be regenerated and re-checked against `paper.md`'s claims before this is presented as current.

---

## 11. Model B's published numbers do not reproduce — OPEN, not fixed

**Files:** `README.md` line 78, `paper.md` lines 9/132/135, `session.md` line 73

**Problem:** Models A and C match their published figures exactly. Model B does not:

| | Published | Actual (`eval_run.py`) |
|---|---|---|
| B accuracy | 84.95% ± **6.01%** | 84.95% ± **5.88%** |
| B macro F1 | **0.8424** ± 0.0590 | **0.8415** ± 0.0580 |

Mean accuracy matches; the standard deviation and macro F1 do not.

**Not caused by the audit branch** — verified two ways. First, nothing changed on this branch is in Model B's code path (`git diff --name-only` touches no `loader.py`, `fuzzy_grader.py`, `model_a.py`, or `model_b.py`; the only `base.py` edit is inside `get_catboost_params_c()`, which Model B never calls). Second, Model B's entire path has been frozen since commit `b6e6516` (2026-06-22), while the paper numbers were written in `290446a` (2026-07-12) — the code was already static when they were published. Most likely a transcription error or a library-version shift.

**Why it matters:** the abstract's headline **"44% reduction in prediction variance"** is computed from the stale ±6.01% (1 − 3.34/6.01 = 44.4%). Against the real ±5.88%, it is **43.3%**. A reviewer re-running `eval_run.py` gets a different number than the abstract claims.

**Fix (not applied):** re-run `eval_run.py`, update B's row in all four locations, and reword the "44%" claim to 43%. Left for Ridanshi — it changes a headline claim in her paper.

---

## 12. Abstract claims per-class F1 improved on all six diseases — it is five, OPEN, not fixed

**File:** `paper.md` line 9 (abstract)

**Problem:** the abstract states the system "improves per-class F1 for all six diseases." Measured B vs C:

```
psoriasis                    B=0.9364  C=0.9459  +0.0095  IMPROVED
seborrheic_dermatitis        B=0.6772  C=0.7619  +0.0847  IMPROVED
lichen_planus                B=0.9787  C=0.9787  +0.0000  EQUAL      <-- not improved
pityriasis_rosea             B=0.7692  C=0.8511  +0.0819  IMPROVED
chronic_dermatitis           B=0.7327  C=0.7963  +0.0636  IMPROVED
pityriasis_rubra_pilaris     B=0.9744  C=0.9756  +0.0012  IMPROVED
```

Lichen planus is **exactly tied**, not improved. The honest claim is "improves per-class F1 on five of six diseases, with the sixth unchanged" — which is still a strong result and costs nothing to state correctly.

Also in the same paragraph: `paper.md` line 135 says the system "improves macro F1 by 0.0426." The actual delta is **0.0436** (0.8850 − 0.8415).

**Fix (not applied):** correct both statements. Unambiguous factual errors, but they live in Ridanshi's paper.

---

## 13. Cosmetic leftovers — deliberately NOT fixed

Two dead-code items were found and left alone on purpose, recorded here so nobody "discovers" them again and assumes they were missed.

**`catboost_trial.py` lines 91–92** still hardcode `od_type="Iter"` / `od_wait=30`. This is the same dead config as finding #9, but the fix could not reach it: the script never calls `get_catboost_params_c()` — it defines its own inline CatBoost params (lines 80–93), written that way in the migration commit so the comparison used fixed literals. It is equally inert there (line 40 is `model.fit(X_train, y_train)`, no `eval_set`), so removing the two lines would not change the script's output at all. Left as a historical reproduction of what was actually run at migration time. Its comment `# XGBoost (current Model C baseline)` is also stale for the same reason.

**`src/grading/fuzzy_grader.py` lines 5–9** define a `BINARY_FEATURES` constant that is never referenced — `grade_series()` correctly leaves binary features untransformed without consulting it. Harmless documentation-by-constant.

---

## 14. Test coverage added for the audit's own changes

**File:** `tests/test_triage.py`, `tests/test_feature_engineer.py`

**Problem found:** the fixes for #3, #5 and #7 shipped with no test coverage. The suite passed 43/43 whether or not those fixes were present — `contradiction_severity`'s `=0.0` default meant the new safety gate was never exercised, and no test asserted `inflammation_burden` or `scale_erythema_ratio` values at all. Silent regressions were possible.

**Added 5 tests (43 → 48 passing):**
- `test_contradiction_blocks_safe` — high contradiction vetoes `SAFE_BIOPSY_FREE`
- `test_contradiction_defaults_to_permissive` — omitting the param must not silently block
- `test_batch_recommend_reads_contradiction_column` — `batch_recommend()` actually consults the column
- `test_inflammation_burden_spans_full_range` — pins the `/3.0` divisor
- `test_scale_erythema_ratio_is_clipped` — pins the 3.0 clip

**Mutation-verified, not assumed:** reintroducing all three original bugs makes exactly 4 of the 5 new tests fail (`4 failed, 44 passed`); the files were then restored and confirmed byte-identical. The tests genuinely discriminate.

---

## Not included here

Anything that's a specific design-choice *value* rather than a bug, doc-drift, or unvalidated-methodology gap (e.g. exact FSM thresholds, individual D-tier weight numbers, biopsy triage cutoffs) is not listed — those reflect clinical judgment calls Ridanshi made deliberately, not something to "fix" without discussing intent first. (Finding #6 above flags the *absence of a validation process* for these values, not any specific value being wrong.)
