# Session Handoff — Hritwik's Understanding & Audit Track

**Purpose of this file**: context for starting a fresh session picking up this exact effort. Read this + `architecture.md` + `changes-made.md` to get fully up to speed. This file is the "what/why/who," the other two are the "how deep."

> **⚠️ SUPERSEDED IN PART — read `changes-made.md` first.** This file documents the **audit phase**, when the rule was strictly read-only. That rule no longer holds: a later implementation phase applied fixes for 8 of the findings on branch `hritwik/audit-fixes`, with Hritwik's explicit sign-off. Statements below asserting that nothing in `src/` was modified describe the audit phase only, and are no longer true of the current branch. `changes-made.md` is the authoritative record of what was actually changed.

---

## Who's asking, and what this track actually is

Hritwik (23BAI0054) is a teammate on this project — **Ridanshi Agarwal (23BCI0026) built the entire HSCIS-ESD codebase**; Hritwik did not write any of it. This session's whole purpose has been **deep understanding and independent audit**, not implementation. Nothing in `src/`, `rules/`, or any pipeline script has been modified — confirmed via `git status`, only new files were created (this file, `architecture.md`, `changes-made.md`, the review deck). That boundary is deliberate: findings get flagged for Ridanshi's review, not silently fixed, since it's her authored work and her call on intent.

Course context: **BCSE 497J – Project-I**, faculty guide **Dr. Gunavathi C**. First review presentation already built this session (see below).

## How this session likes to work — read before continuing

A feedback memory already exists at `/Users/hritwik/.claude/projects/-Users-hritwik-ML-proj-esd-nsai/memory/feedback_explanation_style.md` — a new session should load it, but the short version:

- **Purpose before mechanism, always.** State why something exists before how it works. Getting this backwards was explicitly called out as annoying, more than once.
- **One step at a time, then stop and let the user say "continue."** Don't chain multiple concepts into one dense message — this was explicitly flagged ("you are skipping steps") when it happened.
- **Verify with real code, don't assert from memory.** Nearly every claim in `architecture.md` and `changes-made.md` is backed by an actual script run against the real 366-patient dataset, not textbook reasoning alone. Several assumptions going in turned out wrong once tested (an "obviously fine" unbounded feature had zero accuracy impact; a value assumed unused turned out to rank #11/29 in importance; a speculative concern about the hyperparameter sweep turned out unfounded once actually tested).
- **Don't soften a finding's severity just because one empirical test came back neutral.** Report the test honestly, keep the surrounding reasoning at full weight.
- Hritwik is learning this codebase from scratch — patient, checked-in pacing is the right default, not density.

## Concrete technical facts worth carrying over (not written anywhere else)

- A single running patient example was used throughout the deep-dive for continuity: `erythema=0.667, scaling=0.667, definite_borders=0.333, itching=0.333, koebner=1.0, knee_elbow=1.0, scalp=1.0, family_history=1.0, polygonal=0.0, follicular=0.0, oral_mucosal=0.0, age=0.4375` — traced end-to-end (psoriasis certainty 0.7317 → CatBoost confidence 0.9448 → BiopsyTriage verdict `UNCERTAIN`). Reuse it if picking a stage back up for continuity, or start fresh with a new patient if that's clearer.
- Local machine needed two one-time fixes to actually run the pipeline: `pip3 install ucimlrepo catboost certifi`, and every dataset-fetching script needs `SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")` prefixed (macOS python.org build's cert bundle isn't wired up by default). Worth doing once at the start of a new session rather than rediscovering it.
- `catboost_info/` regenerates automatically every time any script calls `CatBoostClassifier().fit()` — harmless training-log clutter, safe to `rm -rf` freely, not a tracked file.

---

## What's been covered — full pipeline, Stages 1-6, all in `architecture.md`

`architecture.md` is the deep technical reference — read it directly rather than this summary for actual mechanics. High-level map of what's in there:

1. **Raw input** (12 clinical features, why biopsy's 22 histopathological features are excluded)
2. **FuzzyGrader** (fixed-constant min-max scaling, not sklearn's version)
3. **FeatureEngineer** (8 hand-designed features, MI-validated, one dropped)
4. **Symbolic engine** — the deepest section:
   - `RuleEngine`: all 45 rules cataloged, the threshold-gate vs. min-strength mechanic fully worked (SEB_A02 example), why `min` not addition
   - `ConflictAnalyzer`: `conflict_load` (why multiply not sum/min, why pairs not N-way product) and `contradiction_severity` (why max not sum) — both proven with contrasting numeric test cases, not just asserted
   - `DiagnosticFSM`: all 4 gates individually justified, the override mechanism explained, full patient trace
5. **CatBoost classifier** (Stage 5) — purpose vs. symbolic layer alone, real extracted Tree 0, Random Forest vs. gradient boosting vs. AdaBoost distinctions, full hyperparameter table, `StratifiedKFold` proven necessary with real PRP fold-composition data, the prediction-accumulation S-curve traced tree-by-tree, `_cross_validate_c()`'s pooling-vs-averaging design verified sound
6. **BiopsyTriage** (Stage 6) — confirmed to run in *parallel* to CatBoost, not after it (a real structural correction from an earlier draft), fixed thresholds, traced to `UNCERTAIN` for the running patient

## What's explicitly NOT covered yet

- **Evaluation layer** — McNemar test, ablation study, SHAP/imodels explainability (`src/evaluation/`)
- **`app.py`** as its own stage — referenced throughout (its `predict()` function was key evidence for the Stage 5/6 parallel-structure finding), but never walked through top-to-bottom as a subject in itself
- **Models A and B** — explicitly out of scope by Hritwik's own call ("not interested in models a and b as our real selling point is model c")
- Two small Stage 5 loose ends, low priority: `catboost_trial.py`'s actual head-to-head CatBoost-vs-XGBoost numbers (conceptual differences are covered, the specific comparison run isn't), and naming the softmax→argmax step explicitly (seen implicitly via the probability trajectory, never named as its own concept)

---

## `changes-made.md` — 9 verified findings (as of the audit phase; later grew to 14, most now applied)

Quick index (full detail, evidence, and proposed fixes are in the file itself):

1. `requirements.txt` missing `streamlit` + `catboost`
2. `session.md` (Ridanshi's dev log) stale — pre-CatBoost numbers, doesn't mention `app.py`
3. `inflammation_burden` divisor bug (`/9` should be `/3`) — compresses feature to `[0, 0.333]`
4. Rule count documentation drift — docs say 41, actual count is 45 (fixed in the review deck, not yet in `README.md`/`paper.md`/`session.md`)
5. `scale_erythema_ratio` unbounded (real max 33.33 confirmed on the dataset) — empirically zero current accuracy impact, still flagged for interpretability/robustness
6. Rule tier/weight values have no empirical calibration — methodology gap, not a bug (textbook judgment only, unlike the MI-validated engineered features)
7. **`BiopsyTriage` ignores `contradiction_severity`** — most consequential finding. It outranks the two signals (`fsm_state`, `conflict_load`) the actual safety-critical triage decision relies on, confirmed via feature-importance ranking and a real train-with/without accuracy test
8. `rsm` (CatBoost's feature-subsampling regularization) silently dropped during the XGBoost→CatBoost migration; `min_data_in_leaf`/`random_strength` also untested
9. Overfitting detector (`od_type`/`od_wait`) is configured but completely inert — no `eval_set` ever passed to `.fit()`, empirically confirmed zero effect regardless of setting

**One thing tested and explicitly *not* logged**: whether the hyperparameter sweep's accuracy-only selection criterion (`sweep_catboost.py` never tracks macro F1) hides a better-balanced config. Reran the full 108-combo grid tracking accuracy, macro F1, and PRP F1 together — the accuracy-optimal and macro-F1-optimal configs turned out to be **the exact same one**, rank #1/108 on both. Speculation disproven, not a finding.

---

## Other deliverables from this session

- **`Review 1 - 22 07 2026 - filled.pptx`** — the actual Review-I presentation, built from the blank university template (`Review 1 - 22 07 2026.pptx`, left untouched). All 7 slides populated: title/team, aim, abstract (pulled from `paper.md`), a 5-source literature review table, research gap, objectives (with SDG 3 + SDG 10, Conference Paper + Patent as outcomes, TRL 3), and 9 IEEE-formatted references. Uses "45-rule" (corrected), while the source docs (`README.md`/`paper.md`) still say 41 — see finding #4.

## Suggested next steps, in roughly the order they were being considered

1. Evaluation layer (McNemar/ablation/SHAP) or `app.py` as its own stage — whichever Hritwik picks
2. Decide whether to actually run the proposed `rsm`/`min_data_in_leaf`/`random_strength` targeted sweep from finding #8 (would produce numbers that could differ from the currently-published 88.79%, needs review before folding into official results)
3. At some point, hand `changes-made.md` to Ridanshi for a real conversation about which findings to act on — this session has deliberately stopped short of that

**Update (post-audit):** step 2 was done (finding #8 — swept, no gain, no change made) and step 3's implementation half was carried out on branch `hritwik/audit-fixes`. The *conversation* with Ridanshi still hasn't happened — that remains the real next step, and two items (#7's threshold, #6's rule weights) are deliberately parked for her.
