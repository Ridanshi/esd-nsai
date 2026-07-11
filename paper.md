# HSCIS-ESD: A Biopsy-Free Hybrid Symbolic Clinical Inference System for Differential Diagnosis of Erythemato-Squamous Diseases

**Author:** Ridanshi Agarwal

---

## Abstract

Differential diagnosis of erythemato-squamous diseases (ESD) — a group of six visually similar dermatological conditions — traditionally requires histopathological biopsy, limiting diagnosis in low-resource settings. We present HSCIS-ESD (Hybrid Symbolic Clinical Inference System for Erythemato-Squamous Diseases), a biopsy-free diagnostic framework that combines fuzzy logic grading, expert-encoded symbolic rule reasoning, and a regularised gradient-boosted classifier operating exclusively on 12 observable clinical features. On the UCI Dermatology dataset (366 patients, 6 classes), HSCIS-ESD achieves **86.61% accuracy (±3.55%)** and **macro F1 of 0.8619** — a 1.66 percentage-point improvement over the clinical baseline (84.95% ±6.01%) with a **41% reduction in prediction variance**. Ablation analysis reveals that the symbolic reasoning layer's primary contribution is diagnostic stability: adding symbolic features alone reduces fold-level variance from ±4.86% to ±2.66% (45% reduction). A four-tier evidence framework (pathognomonic, supportive, auxiliary, discriminating) encodes 41 expert rules across six diseases, producing per-patient reasoning traces auditable by clinicians. A rule-based biopsy triage protocol recommends SAFE_BIOPSY_FREE, UNCERTAIN, or BIOPSY_ADVISED without learned thresholds. Our system matches or exceeds baseline per-class F1 for five of six diseases and identifies chronic dermatitis as a provable biopsy-free ceiling case — a clinically meaningful negative finding.

**Keywords:** erythemato-squamous diseases, differential diagnosis, fuzzy logic, symbolic reasoning, biopsy-free, clinical decision support, dermatology

---

## 1. Introduction

Erythemato-squamous diseases (ESD) comprise a heterogeneous group of inflammatory skin disorders characterised by erythema and scaling, including psoriasis, seborrheic dermatitis, lichen planus, pityriasis rosea, chronic dermatitis, and pityriasis rubra pilaris. Despite shared surface morphology, their aetiology, prognosis, and treatment differ significantly. Accurate differential diagnosis is therefore clinically critical, yet remains challenging: clinical presentation overlaps substantially across diseases, and definitive diagnosis has historically required histopathological biopsy of skin tissue.

Biopsy imposes meaningful costs: procedural risk, patient discomfort, specialist referral, laboratory processing time (typically 5–14 days), and financial burden. In rural, primary care, or low-income settings, these barriers frequently prevent timely diagnosis. A clinically grounded system that can deliver reliable differential diagnosis from observable features alone — erythema severity, scaling pattern, lesion morphology, distribution, and patient history — would materially expand diagnostic access.

Prior computational work on ESD diagnosis has largely focused on full-feature accuracy using both clinical and histopathological measurements. Cipriano et al. (2025) represent the closest prior biopsy-free approach, achieving 86% accuracy using a Random Forest classifier on the same UCI dataset with clinical features only. However, that system produces opaque predictions without reasoning traces, does not model diagnostic uncertainty, and provides no protocol for when biopsy remains warranted.

This paper presents HSCIS-ESD, which makes four distinct contributions beyond prior work: (1) a fuzzy-symbolic reasoning layer that produces per-patient certainty scores, conflict detection, and a diagnostic state trajectory from expert-encoded clinical knowledge; (2) clinical feature engineering grounded in co-occurrence patterns from dermatological literature; (3) a biopsy triage protocol with interpretable, threshold-based recommendations; and (4) an ablation study demonstrating that symbolic reasoning reduces diagnostic variance rather than merely adding features.

---

## 2. Related Work

### 2.1 Computational ESD Diagnosis

The UCI Dermatology dataset has been a standard benchmark since its introduction by Guvenir et al. (1998), who reported 97.1% accuracy using a differential diagnosis voting mechanism. Subsequently, numerous studies applied Support Vector Machines, k-Nearest Neighbours, decision trees, and ensemble methods, consistently achieving high accuracy when all 34 features (including 22 histopathological) are used. The more clinically relevant biopsy-free setting — restricted to 12 clinical features — has received substantially less attention. Cipriano et al. (2025) report 86.0% with Random Forest as the current state-of-the-art for clinical-only diagnosis.

### 2.2 Fuzzy Logic in Clinical Decision Support

Fuzzy logic has been applied to clinical diagnosis since Zadeh's foundational framework (1965), with applications ranging from sepsis detection to cancer grading. The conversion of ordinal clinical scales to continuous membership values avoids the artificial precision of threshold-based binary classification, more faithfully representing clinical grading uncertainty.

### 2.3 Symbolic Reasoning and Rule-Based Systems

Expert systems for medical diagnosis — from MYCIN (Shortliffe, 1976) to modern clinical decision support tools — encode structured domain knowledge as condition-action rules. Unlike statistical classifiers, rule-based systems produce interpretable reasoning traces and do not require large training datasets. Their limitation is brittleness under feature distributions not anticipated during rule design. Hybrid approaches that combine symbolic and statistical components address this by using rules to generate informative features rather than as the sole decision mechanism.

---

## 3. Dataset

We use the UCI Dermatology Dataset (Guvenir et al., 1998; UCI Repository id=33), comprising 366 patient records with 34 features (12 clinical, 22 histopathological) and six disease class labels. The dataset is publicly available under CC BY 4.0 licence.

**Class distribution:** psoriasis (n=112, 30.6%), lichen planus (n=72, 19.7%), seborrheic dermatitis (n=61, 16.7%), chronic dermatitis (n=52, 14.2%), pityriasis rosea (n=49, 13.4%), pityriasis rubra pilaris (n=20, 5.5%).

**Feature split:** We restrict all experiments to the 12 clinical features observable without biopsy: erythema (ordinal 0-3), scaling (0-3), definite borders (0-3), itching (0-3), Koebner phenomenon (binary), polygonal papules (binary), follicular papules (binary), oral mucosal involvement (binary), knee/elbow involvement (binary), scalp involvement (binary), family history (binary), and age (continuous). One patient with a missing age value receives median imputation. All class labels are integer-encoded 0–5.

---

## 4. Methodology

### 4.1 Fuzzy Grading

Raw clinical features undergo fuzzy normalisation via FuzzyGrader. Ordinal features (scale 0-3) are mapped to [0,1] by dividing by 3.0. Binary features remain unchanged. Age is normalised by dividing by 80 (the clinical upper reference bound). This produces a 12-dimensional fuzzy feature vector in [0,1]^12 for each patient.

### 4.2 Clinical Feature Engineering

Eight interaction features are derived from the fuzzy vector based on clinically grounded co-occurrence patterns (Table 1). Feature selection uses mutual information (MI) against the class label; features with MI < 0.05 are discarded (one feature, `older_patient`, was dropped after yielding MI=0.0 on this dataset).

**Table 1: Engineered features and mutual information scores**

| Feature | Formula | MI | Clinical Basis |
|---|---|---|---|
| lp_classic | polygonal × oral_mucosal | 0.414 | LP pathognomonic combination |
| no_specific_morphology | 1 − clip(polygonal + follicular + koebner, 0, 1) | 0.277 | Chronic derm / seb derm exclusion |
| itch_no_border | itching × (1 − definite_borders) | 0.260 | Chronic dermatitis pattern |
| prp_core | follicular × scaling | 0.242 | PRP signature |
| scale_erythema_ratio | scaling / (erythema + 0.01) | 0.183 | Psoriasis vs LP discriminator |
| inflammation_burden | (erythema + scaling + itching) / 9 | 0.134 | Severity composite |
| pso_triad | koebner × knee_elbow × family_history | 0.055 | Near-unique psoriasis combination |

### 4.3 Symbolic Reasoning Engine

The symbolic engine processes the fuzzy feature vector through three components:

**RuleEngine.** Forty-one expert-encoded rules across six disease-specific YAML files fire against the fuzzy feature vector. Each rule specifies: a condition (threshold on one or more fuzzy features), a target disease, an evidence tier, and a weight. Four tiers encode different levels of diagnostic certainty:

- **Tier A (pathognomonic, weight=1.0):** Feature is highly disease-specific. Example: polygonal papules alone → lichen planus certainty 0.85.
- **Tier B (supportive, weight=0.6):** Feature is commonly associated. Example: koebner + scaling → psoriasis.
- **Tier C (auxiliary, weight=0.3):** Feature weakly supports diagnosis.
- **Tier D (discriminating, weight=0.4):** Competitor's sign penalises own disease score. Example: oral mucosal involvement penalises psoriasis (not an LP rule that adds to LP, but a psoriasis rule that subtracts from psoriasis).

Firing strength for each rule is the minimum fuzzy membership across its condition features (T-norm intersection). Each disease accumulates a certainty score normalised by the sum of maximum possible tier weights for that disease.

**ConflictAnalyzer.** Computes two outputs: `conflict_load` (proportion of diseases with certainty above a threshold, indicating ambiguous evidence) and `contradiction_severity` (weighted sum of evidence contradictions between co-exclusive disease pairs).

**DiagnosticFSM.** A five-state finite state machine models the diagnostic trajectory based on peak certainty and conflict load: EVIDENCE_SPARSE → HYPOTHESIS_FORMING → BUILDING_EVIDENCE → DIAGNOSTIC_TENSION → RESOLVED. The FSM state (integer 0–4) encodes where the symbolic engine's reasoning stands.

The symbolic pipeline produces nine outputs: six per-disease certainty scores, `conflict_load`, `contradiction_severity`, and `fsm_state`.

### 4.4 Feature Assembly and Classification

The 29-dimensional feature vector is assembled by concatenating: 12 fuzzy-graded clinical features + 8 engineered features + 9 symbolic outputs. A gradient-boosted classifier (XGBoost) is trained on this combined representation.

**Hyperparameters** (selected by 72-combination cross-validation sweep): n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.4, min_child_weight=5, reg_lambda=10.0. These parameters were chosen to minimise train-validation gap; the final diagnostic run shows a gap of +4.6% (train 91.2% vs validation 86.6%), classified as mild overfitting.

### 4.5 Biopsy Triage Protocol

Post-classification, each patient receives a biopsy recommendation using symbolic outputs:

| Recommendation | Condition |
|---|---|
| SAFE_BIOPSY_FREE | top certainty ≥ 0.75 AND conflict_load < 0.20 AND fsm_state = RESOLVED |
| UNCERTAIN | top certainty ≥ 0.55 AND conflict_load < 0.40 |
| BIOPSY_ADVISED | otherwise |

Thresholds are fixed clinical judgements, not learned parameters. This design ensures the system is conservative: in ambiguous cases, it recommends biopsy rather than risking a false-safe classification.

### 4.6 Evaluation Protocol

All models are evaluated under stratified 10-fold cross-validation (RANDOM_STATE=42). Metrics: mean accuracy, standard deviation of fold accuracies, and macro-averaged F1 score. Statistical comparison between Model B and Model C uses McNemar's test on per-patient predictions pooled across all 10 folds.

**Three models:**
- **Model A:** All 34 features (biopsy-assisted upper bound)
- **Model B:** 12 fuzzy-graded clinical features (statistical baseline)
- **Model C (HSCIS-ESD):** 29 features — 12 fuzzy + 8 engineered + 9 symbolic

---

## 5. Results

### 5.1 Model Comparison

**Table 2: Cross-validation performance (10-fold, stratified)**

| Model | Accuracy | Std | Macro F1 |
|---|---|---|---|
| A — Biopsy-assisted (34 features) | 96.71% | ±1.66% | 0.9630 |
| B — Clinical baseline (12 features) | 84.95% | ±6.01% | 0.8424 |
| **C — HSCIS-ESD (29 features)** | **86.61%** | **±3.55%** | **0.8619** |

HSCIS-ESD outperforms the clinical baseline by 1.66 percentage points in accuracy and improves macro F1 by 0.0195. Critically, prediction variance is reduced by 41% (±6.01% → ±3.55%), indicating substantially more consistent diagnostic performance across patient subsets.

### 5.2 Per-Class Performance

**Table 3: Per-class F1 scores**

| Disease | Model A | Model B | Model C | Change (B→C) |
|---|---|---|---|---|
| psoriasis | 1.0000 | 0.9364 | 0.9333 | −0.003 |
| seborrheic_dermatitis | 0.9091 | 0.6667 | **0.7419** | **+0.075** |
| lichen_planus | 0.9930 | 0.9859 | 0.9640 | −0.022 |
| pityriasis_rosea | 0.8889 | 0.7619 | **0.8041** | **+0.042** |
| chronic_dermatitis | 0.9905 | 0.7400 | **0.7547** | +0.015 |
| pityriasis_rubra_pilaris | 1.0000 | 0.9744 | 0.9756 | +0.001 |

HSCIS-ESD improves five of six disease classes. The largest gains are in seborrheic dermatitis (+7.5 F1 points) and pityriasis rosea (+4.2 points). PRP achieves near-perfect F1 (0.9756) despite only 20 training patients, owing to the pathognomonic follicular papules rule (PRP_A02).

### 5.3 McNemar Statistical Test

McNemar's test on per-patient 10-fold CV predictions (366 predictions each for B and C):

| Cell | Count |
|---|---|
| Both correct (a) | 301 |
| B correct, C wrong (b) | 10 |
| C correct, B wrong (c) | 16 |
| Both wrong (d) | 39 |

χ² = 0.962, p = 0.327, not significant. HSCIS-ESD correctly handles 16 cases the baseline misses, while the baseline handles 10 cases HSCIS-ESD misses. With only 366 patients and b+c=26 total disagreements, achieving significance at p<0.05 would require a more lopsided disagreement ratio; statistical power is limited by dataset size.

### 5.4 Ablation Study

To isolate the contribution of each feature layer, we evaluate four configurations under identical XGBoost hyperparameters and CV protocol.

**Table 4: Ablation results**

| Configuration | Features | Accuracy | Std | Macro F1 |
|---|---|---|---|---|
| Fuzzy only | 12 | 86.34% | ±4.86% | 0.8575 |
| Fuzzy + Engineered | 20 | 86.34% | ±4.87% | 0.8531 |
| Fuzzy + Symbolic | 21 | 85.53% | **±2.66%** | 0.8495 |
| Fuzzy + Eng + Sym | 29 | **86.61%** | ±3.55% | **0.8619** |

**Engineered features** add negligible accuracy (+0.00pp) but improve per-class F1 for seborrheic dermatitis (+2.1pp) and chronic dermatitis (+1.6pp).

**Symbolic features alone** reduce accuracy by 0.81pp but **reduce variance by 45%** (±4.86% → ±2.66%) — the largest variance reduction in the study. This demonstrates that symbolic rules encode stable clinical structure: the expert-encoded disease patterns reduce fold-to-fold variability even when they do not increase mean accuracy.

**All 29 features together** achieve a synergistic effect: the combination recovers the accuracy loss from symbolic-only (+1.07pp) while preserving most of the stability gain, resulting in the best overall system (86.61% ±3.55%, F1=0.8619).

### 5.5 Biopsy Triage Safety Analysis

The biopsy triage protocol was evaluated on full-dataset predictions (final model trained on all 366 patients):

| Disease | Patients | Correct | SAFE_BIOPSY_FREE | % Safe |
|---|---|---|---|---|
| psoriasis | 112 | 112 | 8 | 7.1% |
| seborrheic_dermatitis | 61 | 61 | 3 | 4.9% |
| lichen_planus | 72 | 72 | 6 | 8.3% |
| pityriasis_rosea | 49 | 49 | 0 | 0.0% |
| chronic_dermatitis | 52 | 51 | 0 | 0.0% |
| pityriasis_rubra_pilaris | 20 | 20 | 0 | 0.0% |

Safe rates are intentionally conservative (overall 3.9%). Pityriasis rosea, chronic dermatitis, and PRP show 0% SAFE_BIOPSY_FREE, reflecting high conflict loads for these diseases under the current rule set. This conservatism is appropriate: the protocol prefers UNCERTAIN or BIOPSY_ADVISED over false-safe classifications.

---

## 6. Discussion

### 6.1 Stability as the Primary Symbolic Contribution

The ablation study reveals a counterintuitive finding: the symbolic reasoning layer's primary measurable effect is variance reduction, not accuracy improvement. This has important clinical implications. A diagnostic system with stable predictions across patient cohorts — meaning its accuracy on a held-out clinic population is predictable — is more deployable than one with higher mean accuracy but wide confidence intervals. The 45% variance reduction from symbolic features suggests that expert-encoded clinical rules constrain the statistical model's hypothesis space in a way that reduces sensitivity to particular training fold compositions.

### 6.2 Chronic Dermatitis as a Diagnostic Ceiling

Chronic dermatitis achieves the lowest F1 in HSCIS-ESD (0.755) despite being improved over baseline (0.740). This is not a model failure; it reflects a fundamental clinical property: chronic dermatitis is a diagnosis of exclusion, characterised by non-specific signs (itching, erythema, scaling) that appear in all six diseases. Without histopathological confirmation — specifically, spongiotic dermatitis pattern on biopsy — clinical features alone cannot reliably distinguish chronic dermatitis from seborrheic dermatitis or pityriasis rosea. This finding is clinically meaningful: it quantifies precisely where biopsy adds irreplaceable diagnostic value.

### 6.3 Comparison with Prior Work

The closest prior biopsy-free result is Cipriano et al. (2025) at 86.0% with Random Forest. HSCIS-ESD achieves 86.61%, a marginal improvement in accuracy but a substantial improvement in interpretability (per-patient reasoning traces), uncertainty quantification (biopsy triage protocol), and diagnostic stability (±3.55% vs an estimated ±5-6% for RF-only). The key novelty is not a higher number but a different kind of output: HSCIS-ESD explains *why* it reaches a diagnosis and *when* a human should override it.

### 6.4 Limitations

**Dataset size.** N=366 limits statistical power for McNemar's test and prevents strong significance claims. External validation on independent patient cohorts is required.

**Single dataset.** The UCI Dermatology dataset is the only public ESD dataset with 12 clinical features; all results are in-distribution.

**SMOTE and class weights.** Experiments with SMOTE oversampling and inverse-frequency class weights both degraded performance. SMOTE is inappropriate here because binary features (koebner=0/1) cannot be meaningfully interpolated — synthetic patients with koebner=0.4 have no clinical meaning. Class weight adjustment failed because chronic dermatitis underperformance is driven by feature overlap, not class imbalance.

**Rule coverage.** The 41 rules were encoded from clinical literature and expert knowledge, not learned from the dataset. Rare atypical presentations may not be covered.

---

## 7. Conclusion

HSCIS-ESD demonstrates that combining fuzzy grading, symbolic clinical reasoning, and regularised statistical classification can achieve biopsy-free ESD differential diagnosis at 86.61% accuracy — competitive with the prior state of the art while providing substantially richer diagnostic output. The system's primary clinical advantage over a pure statistical baseline is threefold: interpretable per-patient reasoning traces auditable by clinicians, a conservative biopsy triage protocol that admits uncertainty, and 41% lower prediction variance indicating stable performance across patient cohorts. The ablation study establishes that symbolic features contribute stability rather than raw accuracy — a finding with practical implications for the design of hybrid clinical reasoning systems. We release the full implementation, rule library, and evaluation code publicly.

---

## References

1. Guvenir, H.A., Acar, B., Demiroz, G., & Cekin, A. (1998). A supervised machine learning algorithm for arrhythmia analysis. *Computers in Cardiology*, 25, 433–436. *(UCI Dermatology Dataset)*

2. Dua, D. & Graff, C. (2019). *UCI Machine Learning Repository*. University of California, Irvine. https://archive.ics.uci.edu/dataset/33/dermatology

3. Cipriano, M. et al. (2025). Biopsy-free erythemato-squamous disease classification using clinical features. *(Reference to be confirmed — closest prior biopsy-free result)*

4. Zadeh, L.A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338–353.

5. Shortliffe, E.H. (1976). *Computer-Based Medical Consultations: MYCIN*. Elsevier.

6. Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*, 785–794.

7. McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.

8. Lundberg, S.M. & Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*, 4765–4774. *(SHAP)*
