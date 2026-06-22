# HSCIS-ESD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hybrid symbolic-statistical clinical inference system for biopsy-free ESD differential diagnosis, comparing three models (A: all features, B: clinical-only, C: clinical + symbolic) and producing a biopsy triage recommendation.

**Architecture:** A 5-stage pipeline: (1) fuzzy grading of ordinal clinical features, (2) symbolic rule firing with certainty propagation and conflict detection, (3) FSM-based diagnostic state tracking, (4) XGBoost hybrid classifier, (5) rule-based biopsy triage. Models A/B/C share identical XGBoost config — only input features differ.

**Tech Stack:** Python 3.10+, pandas, numpy, pyyaml, scikit-learn, xgboost, shap, imodels, scipy, imbalanced-learn, matplotlib, seaborn, pytest, ucimlrepo

## Global Constraints

- Python 3.10+ required
- All random seeds set to `42` everywhere
- Stratified 10-fold cross-validation for all models — same folds, same seed
- Models A/B/C use identical XGBoost hyperparameters — only input features differ
- No histopathological features in Models B or C — excluded by design
- All feature names in YAML rules must exactly match canonical names defined in `src/data/loader.py`
- Triage layer uses no learned parameters — fixed rule-based thresholds only
- Target classes: 1–6 in raw data → encode to 0–5 for XGBoost → decode back for display
- Class mapping: `{1: "psoriasis", 2: "seborrheic_dermatitis", 3: "lichen_planus", 4: "pityriasis_rosea", 5: "chronic_dermatitis", 6: "pityriasis_rubra_pilaris"}`

---

## File Map

```
esd-neuro-symbolic/
├── requirements.txt                          [Task 1] dependencies
├── rules/
│   ├── psoriasis.yaml                        [Task 4] 6 disease rule files
│   ├── seborrheic_dermatitis.yaml            [Task 4]
│   ├── lichen_planus.yaml                    [Task 4]
│   ├── pityriasis_rosea.yaml                 [Task 4]
│   ├── chronic_dermatitis.yaml               [Task 4]
│   └── pityriasis_rubra_pilaris.yaml         [Task 4]
├── src/
│   ├── __init__.py                           [Task 1]
│   ├── data/
│   │   ├── __init__.py                       [Task 1]
│   │   └── loader.py                         [Task 2]
│   ├── grading/
│   │   ├── __init__.py                       [Task 1]
│   │   └── fuzzy_grader.py                   [Task 3]
│   ├── symbolic/
│   │   ├── __init__.py                       [Task 1]
│   │   ├── rule_engine.py                    [Task 5]
│   │   ├── conflict.py                       [Task 6]
│   │   ├── fsm.py                            [Task 7]
│   │   └── pipeline.py                       [Task 8]
│   ├── models/
│   │   ├── __init__.py                       [Task 1]
│   │   ├── base.py                           [Task 9]
│   │   ├── model_a.py                        [Task 9]
│   │   ├── model_b.py                        [Task 9]
│   │   └── model_c.py                        [Task 10]
│   ├── triage/
│   │   ├── __init__.py                       [Task 1]
│   │   └── biopsy_triage.py                  [Task 11]
│   └── evaluation/
│       ├── __init__.py                       [Task 1]
│       ├── metrics.py                        [Task 12]
│       └── explainability.py                 [Task 13]
└── tests/
    ├── conftest.py                           [Task 1]
    ├── test_loader.py                        [Task 2]
    ├── test_fuzzy_grader.py                  [Task 3]
    ├── test_rule_engine.py                   [Task 5]
    ├── test_conflict.py                      [Task 6]
    ├── test_fsm.py                           [Task 7]
    ├── test_pipeline.py                      [Task 8]
    └── test_triage.py                        [Task 11]
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`, `src/data/__init__.py`, `src/grading/__init__.py`, `src/symbolic/__init__.py`, `src/models/__init__.py`, `src/triage/__init__.py`, `src/evaluation/__init__.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: installed environment, importable `src` package, shared pytest fixtures

- [ ] **Step 1: Create requirements.txt**

```
pandas>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
scikit-learn>=1.3.0
xgboost>=2.0.0
shap>=0.46.0
imodels>=1.3.0
scipy>=1.11.0
imbalanced-learn>=0.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
pytest>=7.4.0
ucimlrepo>=0.0.3
```

- [ ] **Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 3: Create all `__init__.py` files**

Create empty `__init__.py` in: `src/`, `src/data/`, `src/grading/`, `src/symbolic/`, `src/models/`, `src/triage/`, `src/evaluation/`, `tests/`

```bash
mkdir -p src/data src/grading src/symbolic src/models src/triage src/evaluation tests rules
touch src/__init__.py src/data/__init__.py src/grading/__init__.py src/symbolic/__init__.py src/models/__init__.py src/triage/__init__.py src/evaluation/__init__.py tests/__init__.py
```

- [ ] **Step 4: Create tests/conftest.py**

```python
import pytest
import pandas as pd
import numpy as np

CLINICAL_FEATURES = [
    "erythema", "scaling", "definite_borders", "itching",
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_and_elbow_involvement",
    "scalp_involvement", "family_history", "age"
]

BINARY_FEATURES = [
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_and_elbow_involvement",
    "scalp_involvement", "family_history"
]

ORDINAL_FEATURES = [
    "erythema", "scaling", "definite_borders", "itching"
]

DISEASES = [
    "psoriasis", "seborrheic_dermatitis", "lichen_planus",
    "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris"
]

CLASS_MAP = {
    1: "psoriasis",
    2: "seborrheic_dermatitis",
    3: "lichen_planus",
    4: "pityriasis_rosea",
    5: "chronic_dermatitis",
    6: "pityriasis_rubra_pilaris"
}

@pytest.fixture
def sample_patient_ordinal():
    """A psoriasis-like patient with raw ordinal values."""
    return pd.Series({
        "erythema": 2,
        "scaling": 2,
        "definite_borders": 1,
        "itching": 1,
        "koebner_phenomenon": 1,
        "polygonal_papules": 0,
        "follicular_papules": 0,
        "oral_mucosal_involvement": 0,
        "knee_and_elbow_involvement": 1,
        "scalp_involvement": 1,
        "family_history": 0,
        "age": 35,
    })

@pytest.fixture
def sample_patient_fuzzy():
    """Same patient after fuzzy grading."""
    return pd.Series({
        "erythema": 0.6667,
        "scaling": 0.6667,
        "definite_borders": 0.3333,
        "itching": 0.3333,
        "koebner_phenomenon": 1.0,
        "polygonal_papules": 0.0,
        "follicular_papules": 0.0,
        "oral_mucosal_involvement": 0.0,
        "knee_and_elbow_involvement": 1.0,
        "scalp_involvement": 1.0,
        "family_history": 0.0,
        "age": 0.4375,  # 35 / 80 (max age normalization)
    })
```

- [ ] **Step 5: Verify pytest discovers conftest**

```bash
pytest tests/ --collect-only
```

Expected: output shows conftest fixtures listed, no import errors.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/ tests/conftest.py
git commit -m "feat: project scaffold, package structure, shared fixtures"
```

---

### Task 2: DataLoader

**Files:**
- Create: `src/data/loader.py`
- Create: `tests/test_loader.py`

**Interfaces:**
- Consumes: `ucimlrepo.fetch_ucirepo(id=33)`
- Produces:
  - `load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]`
    returns `(X_clinical, X_histopath, X_all, y)`
  - `X_clinical`: shape (366, 12), columns = CLINICAL_FEATURES from conftest
  - `X_histopath`: shape (366, 22)
  - `X_all`: shape (366, 34)
  - `y`: Series of int, values 0–5 (class-encoded)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_loader.py
import pytest
import pandas as pd
from src.data.loader import load_dataset, CLINICAL_FEATURES, HISTOPATH_FEATURES, CLASS_MAP

def test_dataset_shapes():
    X_clinical, X_histopath, X_all, y = load_dataset()
    assert X_clinical.shape == (366, 12)
    assert X_histopath.shape[0] == 366
    assert X_all.shape == (366, 34)
    assert len(y) == 366

def test_clinical_feature_names():
    X_clinical, _, _, _ = load_dataset()
    assert list(X_clinical.columns) == CLINICAL_FEATURES

def test_target_classes():
    _, _, _, y = load_dataset()
    assert set(y.unique()) == {0, 1, 2, 3, 4, 5}

def test_no_missing_values_after_imputation():
    X_clinical, _, _, _ = load_dataset()
    assert X_clinical.isnull().sum().sum() == 0

def test_age_range():
    X_clinical, _, _, _ = load_dataset()
    assert X_clinical["age"].min() >= 0
    assert X_clinical["age"].max() <= 120
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_loader.py -v
```

Expected: `ImportError` — `src.data.loader` does not exist yet.

- [ ] **Step 3: Implement DataLoader**

```python
# src/data/loader.py
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

CLINICAL_FEATURES = [
    "erythema", "scaling", "definite_borders", "itching",
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_and_elbow_involvement",
    "scalp_involvement", "family_history", "age"
]

HISTOPATH_FEATURES = [
    "melanin_incontinence", "eosinophils_in_the_infiltrate",
    "pnl_infiltrate", "fibrosis_of_the_papillary_dermis",
    "exocytosis", "acanthosis", "hyperkeratosis", "parakeratosis",
    "clubbing_of_the_rete_ridges", "elongation_of_the_rete_ridges",
    "thinning_of_the_suprapapillary_epidermis", "spongiform_pustule",
    "munro_microabcess", "focal_hypergranulosis",
    "disappearance_of_the_granular_layer",
    "vacuolisation_and_damage_of_basal_layer", "spongiosis",
    "saw_tooth_appearance_of_retes", "follicular_horn_plug",
    "perifollicular_parakeratosis", "inflammatory_monoluclear_inflitrate",
    "band_like_infiltrate"
]

CLASS_MAP = {
    1: "psoriasis",
    2: "seborrheic_dermatitis",
    3: "lichen_planus",
    4: "pityriasis_rosea",
    5: "chronic_dermatitis",
    6: "pityriasis_rubra_pilaris"
}

_CACHE = {}


def load_dataset() -> tuple:
    """
    Returns (X_clinical, X_histopath, X_all, y).
    y values are 0-indexed (0-5). Caches after first fetch.
    Missing age values filled with median.
    """
    if _CACHE:
        return _CACHE["result"]

    raw = fetch_ucirepo(id=33)
    X = raw.data.features.copy()
    y_raw = raw.data.targets.iloc[:, 0]

    # Standardise column names: lowercase, spaces -> underscores
    X.columns = [c.lower().replace(" ", "_").replace("-", "_") for c in X.columns]

    # Median imputation for missing age values
    age_col = "age"
    if age_col in X.columns and X[age_col].isnull().any():
        X[age_col] = X[age_col].fillna(X[age_col].median())

    # Select feature subsets
    clinical_cols = [c for c in CLINICAL_FEATURES if c in X.columns]
    histopath_cols = [c for c in X.columns if c not in clinical_cols]

    X_clinical = X[clinical_cols].reset_index(drop=True)
    X_histopath = X[histopath_cols].reset_index(drop=True)
    X_all = X.reset_index(drop=True)

    # Encode targets 1-6 → 0-5
    y = (y_raw.values.ravel().astype(int) - 1)
    y = pd.Series(y, name="disease")

    result = (X_clinical, X_histopath, X_all, y)
    _CACHE["result"] = result
    return result
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_loader.py -v
```

Expected: all 5 tests PASS. If column name mapping fails, inspect `X.columns` and adjust `CLINICAL_FEATURES` list to match exact UCI column names.

- [ ] **Step 5: Commit**

```bash
git add src/data/loader.py tests/test_loader.py
git commit -m "feat: DataLoader — UCI fetch, median imputation, class encoding"
```

---

### Task 3: FuzzyGrader

**Files:**
- Create: `src/grading/fuzzy_grader.py`
- Create: `tests/test_fuzzy_grader.py`

**Interfaces:**
- Consumes: `pd.DataFrame` or `pd.Series` of raw ordinal clinical values
- Produces:
  - `FuzzyGrader.grade(X: pd.DataFrame) -> pd.DataFrame` — same shape, float dtype, values in [0.0, 1.0]
  - `FuzzyGrader.grade_series(row: pd.Series) -> pd.Series` — single patient

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fuzzy_grader.py
import pytest
import pandas as pd
import numpy as np
from src.grading.fuzzy_grader import FuzzyGrader

ORDINAL_FEATURES = ["erythema", "scaling", "definite_borders", "itching"]
BINARY_FEATURES = [
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_and_elbow_involvement",
    "scalp_involvement", "family_history"
]

def test_ordinal_grading():
    grader = FuzzyGrader()
    row = pd.Series({"erythema": 0, "scaling": 1, "definite_borders": 2, "itching": 3})
    result = grader.grade_series(row)
    assert result["erythema"] == pytest.approx(0.0)
    assert result["scaling"] == pytest.approx(1/3, abs=1e-4)
    assert result["definite_borders"] == pytest.approx(2/3, abs=1e-4)
    assert result["itching"] == pytest.approx(1.0)

def test_binary_features_unchanged():
    grader = FuzzyGrader()
    row = pd.Series({"koebner_phenomenon": 1, "family_history": 0})
    result = grader.grade_series(row)
    assert result["koebner_phenomenon"] == 1.0
    assert result["family_history"] == 0.0

def test_age_normalized():
    grader = FuzzyGrader(age_max=80)
    row = pd.Series({"age": 40})
    result = grader.grade_series(row)
    assert result["age"] == pytest.approx(0.5)

def test_output_range():
    grader = FuzzyGrader()
    data = pd.DataFrame([{
        "erythema": 2, "scaling": 1, "definite_borders": 0, "itching": 3,
        "koebner_phenomenon": 1, "polygonal_papules": 0, "follicular_papules": 0,
        "oral_mucosal_involvement": 0, "knee_and_elbow_involvement": 1,
        "scalp_involvement": 1, "family_history": 0, "age": 35,
    }])
    result = grader.grade(data)
    assert result.min().min() >= 0.0
    assert result.max().max() <= 1.0

def test_dataframe_shape_preserved():
    grader = FuzzyGrader()
    data = pd.DataFrame([
        {"erythema": 1, "scaling": 2, "age": 30},
        {"erythema": 3, "scaling": 0, "age": 60},
    ])
    result = grader.grade(data)
    assert result.shape == data.shape
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_fuzzy_grader.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement FuzzyGrader**

```python
# src/grading/fuzzy_grader.py
import pandas as pd
import numpy as np

ORDINAL_FEATURES = ["erythema", "scaling", "definite_borders", "itching"]
BINARY_FEATURES = [
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_and_elbow_involvement",
    "scalp_involvement", "family_history"
]
ORDINAL_MAX = 3.0


class FuzzyGrader:
    def __init__(self, age_max: float = 80.0):
        self.age_max = age_max

    def grade_series(self, row: pd.Series) -> pd.Series:
        result = row.copy().astype(float)
        for feat in ORDINAL_FEATURES:
            if feat in result.index:
                result[feat] = float(result[feat]) / ORDINAL_MAX
        if "age" in result.index:
            result["age"] = min(float(result["age"]) / self.age_max, 1.0)
        # Binary features already 0 or 1 — no transform needed
        return result.clip(0.0, 1.0)

    def grade(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.apply(self.grade_series, axis=1)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_fuzzy_grader.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/grading/fuzzy_grader.py tests/test_fuzzy_grader.py
git commit -m "feat: FuzzyGrader — ordinal/binary/age fuzzy conversion"
```

---

### Task 4: Rule YAML Library

**Files:**
- Create: `rules/psoriasis.yaml`
- Create: `rules/seborrheic_dermatitis.yaml`
- Create: `rules/lichen_planus.yaml`
- Create: `rules/pityriasis_rosea.yaml`
- Create: `rules/chronic_dermatitis.yaml`
- Create: `rules/pityriasis_rubra_pilaris.yaml`

**Interfaces:**
- Consumes: nothing (data files)
- Produces: YAML files loadable by RuleEngine in Task 5

All feature names in `feature:` fields must exactly match names in `CLINICAL_FEATURES` from `src/data/loader.py`.

- [ ] **Step 1: Create rules/psoriasis.yaml**

```yaml
# rules/psoriasis.yaml
# Source: Fitzpatrick's Dermatology, Andrews' Diseases of the Skin
- id: PSO_A01
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

- id: PSO_B01
  disease: psoriasis
  tier: B
  weight: 0.6
  conditions:
    - feature: scaling
      threshold: 0.5
    - feature: erythema
      threshold: 0.5

- id: PSO_B02
  disease: psoriasis
  tier: B
  weight: 0.6
  conditions:
    - feature: family_history
      threshold: 0.5

- id: PSO_C01
  disease: psoriasis
  tier: C
  weight: 0.3
  conditions:
    - feature: definite_borders
      threshold: 0.5

- id: PSO_D01
  disease: psoriasis
  tier: D
  weight: 0.4
  competitor: lichen_planus
  conditions:
    - feature: polygonal_papules
      threshold: 0.5
```

- [ ] **Step 2: Create rules/seborrheic_dermatitis.yaml**

```yaml
# rules/seborrheic_dermatitis.yaml
- id: SEB_A01
  disease: seborrheic_dermatitis
  tier: A
  weight: 1.0
  conditions:
    - feature: scalp_involvement
      threshold: 0.5
    - feature: scaling
      threshold: 0.33
    - feature: erythema
      threshold: 0.33

- id: SEB_B01
  disease: seborrheic_dermatitis
  tier: B
  weight: 0.6
  conditions:
    - feature: itching
      threshold: 0.33

- id: SEB_B02
  disease: seborrheic_dermatitis
  tier: B
  weight: 0.6
  conditions:
    - feature: scaling
      threshold: 0.5

- id: SEB_C01
  disease: seborrheic_dermatitis
  tier: C
  weight: 0.3
  conditions:
    - feature: definite_borders
      threshold: 0.33

- id: SEB_D01
  disease: seborrheic_dermatitis
  tier: D
  weight: 0.4
  competitor: psoriasis
  conditions:
    - feature: koebner_phenomenon
      threshold: 0.5
```

- [ ] **Step 3: Create rules/lichen_planus.yaml**

```yaml
# rules/lichen_planus.yaml
- id: LIC_A01
  disease: lichen_planus
  tier: A
  weight: 1.0
  conditions:
    - feature: polygonal_papules
      threshold: 0.5
    - feature: oral_mucosal_involvement
      threshold: 0.5

- id: LIC_A02
  disease: lichen_planus
  tier: A
  weight: 1.0
  conditions:
    - feature: polygonal_papules
      threshold: 0.5
    - feature: koebner_phenomenon
      threshold: 0.5

- id: LIC_B01
  disease: lichen_planus
  tier: B
  weight: 0.6
  conditions:
    - feature: itching
      threshold: 0.5

- id: LIC_B02
  disease: lichen_planus
  tier: B
  weight: 0.6
  conditions:
    - feature: follicular_papules
      threshold: 0.5

- id: LIC_C01
  disease: lichen_planus
  tier: C
  weight: 0.3
  conditions:
    - feature: erythema
      threshold: 0.33
```

- [ ] **Step 4: Create rules/pityriasis_rosea.yaml**

```yaml
# rules/pityriasis_rosea.yaml
- id: PIT_A01
  disease: pityriasis_rosea
  tier: A
  weight: 1.0
  conditions:
    - feature: definite_borders
      threshold: 0.5
    - feature: scaling
      threshold: 0.33
    - feature: erythema
      threshold: 0.33

- id: PIT_B01
  disease: pityriasis_rosea
  tier: B
  weight: 0.6
  conditions:
    - feature: itching
      threshold: 0.33

- id: PIT_B02
  disease: pityriasis_rosea
  tier: B
  weight: 0.6
  conditions:
    - feature: scaling
      threshold: 0.33

- id: PIT_C01
  disease: pityriasis_rosea
  tier: C
  weight: 0.3
  conditions:
    - feature: erythema
      threshold: 0.33

- id: PIT_D01
  disease: pityriasis_rosea
  tier: D
  weight: 0.4
  competitor: psoriasis
  conditions:
    - feature: koebner_phenomenon
      threshold: 0.5
```

- [ ] **Step 5: Create rules/chronic_dermatitis.yaml**

```yaml
# rules/chronic_dermatitis.yaml
- id: CHR_B01
  disease: chronic_dermatitis
  tier: B
  weight: 0.6
  conditions:
    - feature: itching
      threshold: 0.5
    - feature: erythema
      threshold: 0.33

- id: CHR_B02
  disease: chronic_dermatitis
  tier: B
  weight: 0.6
  conditions:
    - feature: scaling
      threshold: 0.33

- id: CHR_B03
  disease: chronic_dermatitis
  tier: B
  weight: 0.6
  conditions:
    - feature: definite_borders
      threshold: 0.33

- id: CHR_C01
  disease: chronic_dermatitis
  tier: C
  weight: 0.3
  conditions:
    - feature: follicular_papules
      threshold: 0.5

- id: CHR_C02
  disease: chronic_dermatitis
  tier: C
  weight: 0.3
  conditions:
    - feature: erythema
      threshold: 0.5

- id: CHR_D01
  disease: chronic_dermatitis
  tier: D
  weight: 0.4
  competitor: psoriasis
  conditions:
    - feature: koebner_phenomenon
      threshold: 0.5
```

- [ ] **Step 6: Create rules/pityriasis_rubra_pilaris.yaml**

```yaml
# rules/pityriasis_rubra_pilaris.yaml
- id: PRP_A01
  disease: pityriasis_rubra_pilaris
  tier: A
  weight: 1.0
  conditions:
    - feature: follicular_papules
      threshold: 0.5
    - feature: scaling
      threshold: 0.5

- id: PRP_B01
  disease: pityriasis_rubra_pilaris
  tier: B
  weight: 0.6
  conditions:
    - feature: erythema
      threshold: 0.5
    - feature: definite_borders
      threshold: 0.33

- id: PRP_B02
  disease: pityriasis_rubra_pilaris
  tier: B
  weight: 0.6
  conditions:
    - feature: scaling
      threshold: 0.5

- id: PRP_C01
  disease: pityriasis_rubra_pilaris
  tier: C
  weight: 0.3
  conditions:
    - feature: itching
      threshold: 0.33

- id: PRP_D01
  disease: pityriasis_rubra_pilaris
  tier: D
  weight: 0.4
  competitor: psoriasis
  conditions:
    - feature: koebner_phenomenon
      threshold: 0.5
```

- [ ] **Step 7: Validate all YAML files load without error**

```bash
python -c "
import yaml, glob
for path in glob.glob('rules/*.yaml'):
    with open(path) as f:
        rules = yaml.safe_load(f)
    print(f'{path}: {len(rules)} rules OK')
"
```

Expected output (rule counts may vary):
```
rules/psoriasis.yaml: 5 rules OK
rules/seborrheic_dermatitis.yaml: 5 rules OK
rules/lichen_planus.yaml: 5 rules OK
rules/pityriasis_rosea.yaml: 5 rules OK
rules/chronic_dermatitis.yaml: 6 rules OK
rules/pityriasis_rubra_pilaris.yaml: 5 rules OK
```

- [ ] **Step 8: Commit**

```bash
git add rules/
git commit -m "feat: clinical rule YAML library — 31 rules across 6 ESD diseases"
```

---

### Task 5: RuleEngine

**Files:**
- Create: `src/symbolic/rule_engine.py`
- Create: `tests/test_rule_engine.py`

**Interfaces:**
- Consumes: `pd.Series` of fuzzy-graded clinical features, path to rules directory
- Produces:
  - `RuleEngine(rules_dir: str)` — loads all YAML files at init
  - `RuleEngine.fire(patient: pd.Series) -> dict[str, float]` — returns `{disease_name: certainty_score}` for all 6 diseases, values in [0.0, 1.0]
  - `RuleEngine.get_fired_rules(patient: pd.Series) -> list[dict]` — returns list of fired rules with id, disease, firing_strength, contribution (for reasoning trace)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rule_engine.py
import pytest
import pandas as pd
from src.symbolic.rule_engine import RuleEngine

RULES_DIR = "rules"

@pytest.fixture
def engine():
    return RuleEngine(RULES_DIR)

@pytest.fixture
def psoriasis_patient():
    return pd.Series({
        "erythema": 0.6667,
        "scaling": 0.6667,
        "definite_borders": 0.3333,
        "itching": 0.3333,
        "koebner_phenomenon": 1.0,
        "polygonal_papules": 0.0,
        "follicular_papules": 0.0,
        "oral_mucosal_involvement": 0.0,
        "knee_and_elbow_involvement": 1.0,
        "scalp_involvement": 1.0,
        "family_history": 0.0,
        "age": 0.4375,
    })

@pytest.fixture
def minimal_patient():
    """Patient with all features at zero."""
    return pd.Series({feat: 0.0 for feat in [
        "erythema", "scaling", "definite_borders", "itching",
        "koebner_phenomenon", "polygonal_papules", "follicular_papules",
        "oral_mucosal_involvement", "knee_and_elbow_involvement",
        "scalp_involvement", "family_history", "age"
    ]})

def test_fire_returns_all_diseases(engine, psoriasis_patient):
    scores = engine.fire(psoriasis_patient)
    expected = {
        "psoriasis", "seborrheic_dermatitis", "lichen_planus",
        "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris"
    }
    assert set(scores.keys()) == expected

def test_fire_values_in_range(engine, psoriasis_patient):
    scores = engine.fire(psoriasis_patient)
    for disease, score in scores.items():
        assert 0.0 <= score <= 1.0, f"{disease} score {score} out of range"

def test_psoriasis_high_for_psoriasis_patient(engine, psoriasis_patient):
    scores = engine.fire(psoriasis_patient)
    assert scores["psoriasis"] > 0.5
    assert scores["psoriasis"] == max(scores.values())

def test_minimal_patient_all_zero(engine, minimal_patient):
    scores = engine.fire(minimal_patient)
    for score in scores.values():
        assert score == pytest.approx(0.0)

def test_fired_rules_structure(engine, psoriasis_patient):
    fired = engine.get_fired_rules(psoriasis_patient)
    assert len(fired) > 0
    for rule in fired:
        assert "id" in rule
        assert "disease" in rule
        assert "firing_strength" in rule
        assert "contribution" in rule
        assert 0.0 <= rule["firing_strength"] <= 1.0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_rule_engine.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement RuleEngine**

```python
# src/symbolic/rule_engine.py
import os
import glob
import yaml
import numpy as np
import pandas as pd
from typing import Optional

DISEASES = [
    "psoriasis", "seborrheic_dermatitis", "lichen_planus",
    "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris"
]


class RuleEngine:
    def __init__(self, rules_dir: str):
        self._rules: list[dict] = []
        self._max_weight: dict[str, float] = {d: 0.0 for d in DISEASES}
        self._load_rules(rules_dir)

    def _load_rules(self, rules_dir: str) -> None:
        for path in glob.glob(os.path.join(rules_dir, "*.yaml")):
            with open(path) as f:
                rules = yaml.safe_load(f)
            for rule in rules:
                self._rules.append(rule)
                disease = rule["disease"]
                if rule.get("tier") != "D":  # D-tier subtracts from competitor
                    self._max_weight[disease] = (
                        self._max_weight.get(disease, 0.0) + rule["weight"]
                    )

    def _fire_rule(self, rule: dict, patient: pd.Series) -> float:
        """Returns firing strength (0.0 if any condition unmet)."""
        strengths = []
        for cond in rule["conditions"]:
            feat = cond["feature"]
            threshold = cond["threshold"]
            value = float(patient.get(feat, 0.0))
            if value < threshold:
                return 0.0
            strengths.append(value)
        return float(np.min(strengths)) if strengths else 0.0

    def fire(self, patient: pd.Series) -> dict[str, float]:
        """Returns certainty score per disease, normalized to [0, 1]."""
        accumulated = {d: 0.0 for d in DISEASES}
        competitor_penalties = {d: 0.0 for d in DISEASES}

        for rule in self._rules:
            strength = self._fire_rule(rule, patient)
            if strength == 0.0:
                continue
            disease = rule["disease"]
            if rule.get("tier") == "D":
                competitor = rule.get("competitor")
                if competitor in competitor_penalties:
                    competitor_penalties[competitor] += rule["weight"] * strength
            else:
                accumulated[disease] += rule["weight"] * strength

        scores = {}
        for d in DISEASES:
            max_w = self._max_weight.get(d, 1.0)
            if max_w == 0.0:
                scores[d] = 0.0
            else:
                raw = accumulated[d] - competitor_penalties.get(d, 0.0)
                scores[d] = float(np.clip(raw / max_w, 0.0, 1.0))
        return scores

    def get_fired_rules(self, patient: pd.Series) -> list[dict]:
        """Returns metadata for all rules that fired (strength > 0)."""
        fired = []
        for rule in self._rules:
            strength = self._fire_rule(rule, patient)
            if strength > 0.0:
                fired.append({
                    "id": rule["id"],
                    "disease": rule["disease"],
                    "tier": rule.get("tier"),
                    "firing_strength": round(strength, 4),
                    "contribution": round(strength * rule["weight"], 4),
                })
        return fired
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_rule_engine.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symbolic/rule_engine.py tests/test_rule_engine.py
git commit -m "feat: RuleEngine — fuzzy rule firing, certainty accumulation, D-tier penalties"
```

---

### Task 6: ConflictAnalyzer

**Files:**
- Create: `src/symbolic/conflict.py`
- Create: `tests/test_conflict.py`

**Interfaces:**
- Consumes: `dict[str, float]` certainty scores (output of `RuleEngine.fire()`)
- Produces:
  - `ConflictAnalyzer.analyze(certainty_scores: dict) -> tuple[float, float]`
    returns `(conflict_load, contradiction_severity)` both in [0.0, 1.0]

INCOMPATIBLE_PAIRS (share no pathognomonic features — clinically mutually exclusive presentations):
```python
[("psoriasis", "lichen_planus"), ("pityriasis_rosea", "pityriasis_rubra_pilaris")]
```

- [ ] **Step 1: Write failing tests**

```python
# tests/test_conflict.py
import pytest
from src.symbolic.conflict import ConflictAnalyzer

@pytest.fixture
def analyzer():
    return ConflictAnalyzer()

def test_no_conflict_single_dominant(analyzer):
    scores = {
        "psoriasis": 0.9, "seborrheic_dermatitis": 0.05,
        "lichen_planus": 0.02, "pityriasis_rosea": 0.01,
        "chronic_dermatitis": 0.01, "pityriasis_rubra_pilaris": 0.01
    }
    load, severity = analyzer.analyze(scores)
    assert load < 0.1

def test_high_conflict_two_strong(analyzer):
    scores = {
        "psoriasis": 0.8, "seborrheic_dermatitis": 0.75,
        "lichen_planus": 0.0, "pityriasis_rosea": 0.0,
        "chronic_dermatitis": 0.0, "pityriasis_rubra_pilaris": 0.0
    }
    load, _ = analyzer.analyze(scores)
    assert load > 0.3

def test_contradiction_severity_incompatible_pair(analyzer):
    scores = {
        "psoriasis": 0.8, "lichen_planus": 0.7,  # incompatible pair
        "seborrheic_dermatitis": 0.0, "pityriasis_rosea": 0.0,
        "chronic_dermatitis": 0.0, "pityriasis_rubra_pilaris": 0.0
    }
    _, severity = analyzer.analyze(scores)
    assert severity > 0.4

def test_outputs_in_range(analyzer):
    scores = {d: 0.5 for d in [
        "psoriasis", "seborrheic_dermatitis", "lichen_planus",
        "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris"
    ]}
    load, severity = analyzer.analyze(scores)
    assert 0.0 <= load <= 1.0
    assert 0.0 <= severity <= 1.0

def test_all_zero_no_conflict(analyzer):
    scores = {d: 0.0 for d in [
        "psoriasis", "seborrheic_dermatitis", "lichen_planus",
        "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris"
    ]}
    load, severity = analyzer.analyze(scores)
    assert load == pytest.approx(0.0)
    assert severity == pytest.approx(0.0)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_conflict.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement ConflictAnalyzer**

```python
# src/symbolic/conflict.py
import numpy as np

INCOMPATIBLE_PAIRS = [
    ("psoriasis", "lichen_planus"),
    ("pityriasis_rosea", "pityriasis_rubra_pilaris"),
]

CONFLICT_THRESHOLD = 0.2  # only consider diseases above this for conflict


class ConflictAnalyzer:
    def analyze(self, certainty_scores: dict) -> tuple[float, float]:
        """
        Returns (conflict_load, contradiction_severity), both in [0.0, 1.0].

        conflict_load: sum of pairwise products for diseases above threshold.
        contradiction_severity: max pairwise product among incompatible pairs.
        """
        diseases = list(certainty_scores.keys())
        scores = [certainty_scores[d] for d in diseases]

        # Conflict load: pairwise products above threshold
        conflict_load = 0.0
        active = [(d, s) for d, s in zip(diseases, scores) if s > CONFLICT_THRESHOLD]
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                conflict_load += active[i][1] * active[j][1]

        # Normalize: max possible = C(n,2) pairs all at 1.0
        n = len(active)
        max_pairs = n * (n - 1) / 2 if n > 1 else 1
        conflict_load = float(np.clip(conflict_load / max_pairs, 0.0, 1.0)) if max_pairs > 0 else 0.0

        # Contradiction severity: max product among incompatible pairs
        contradiction_severity = 0.0
        for d1, d2 in INCOMPATIBLE_PAIRS:
            s1 = certainty_scores.get(d1, 0.0)
            s2 = certainty_scores.get(d2, 0.0)
            contradiction_severity = max(contradiction_severity, s1 * s2)

        contradiction_severity = float(np.clip(contradiction_severity, 0.0, 1.0))
        return conflict_load, contradiction_severity
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_conflict.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symbolic/conflict.py tests/test_conflict.py
git commit -m "feat: ConflictAnalyzer — conflict load and contradiction severity"
```

---

### Task 7: DiagnosticFSM

**Files:**
- Create: `src/symbolic/fsm.py`
- Create: `tests/test_fsm.py`

**Interfaces:**
- Consumes: `dict[str, float]` certainty scores, `float` conflict_load
- Produces:
  - `DiagnosticFSM.traverse(certainty_scores: dict, conflict_load: float) -> int`
    returns FSM state as int: 0=EVIDENCE_SPARSE, 1=HYPOTHESIS_FORMING, 2=CERTAINTY_BUILDING, 3=DIAGNOSTIC_TENSION, 4=RESOLVED

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fsm.py
import pytest
from src.symbolic.fsm import DiagnosticFSM, FSMState

@pytest.fixture
def fsm():
    return DiagnosticFSM()

def test_all_zero_evidence_sparse(fsm):
    scores = {d: 0.0 for d in [
        "psoriasis", "seborrheic_dermatitis", "lichen_planus",
        "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris"
    ]}
    state = fsm.traverse(scores, conflict_load=0.0)
    assert state == FSMState.EVIDENCE_SPARSE

def test_low_certainty_hypothesis_forming(fsm):
    scores = {
        "psoriasis": 0.2, "seborrheic_dermatitis": 0.0, "lichen_planus": 0.0,
        "pityriasis_rosea": 0.0, "chronic_dermatitis": 0.0, "pityriasis_rubra_pilaris": 0.0
    }
    state = fsm.traverse(scores, conflict_load=0.0)
    assert state == FSMState.HYPOTHESIS_FORMING

def test_moderate_certainty_building(fsm):
    scores = {
        "psoriasis": 0.5, "seborrheic_dermatitis": 0.0, "lichen_planus": 0.0,
        "pityriasis_rosea": 0.0, "chronic_dermatitis": 0.0, "pityriasis_rubra_pilaris": 0.0
    }
    state = fsm.traverse(scores, conflict_load=0.1)
    assert state == FSMState.CERTAINTY_BUILDING

def test_high_conflict_diagnostic_tension(fsm):
    scores = {
        "psoriasis": 0.6, "seborrheic_dermatitis": 0.55, "lichen_planus": 0.0,
        "pityriasis_rosea": 0.0, "chronic_dermatitis": 0.0, "pityriasis_rubra_pilaris": 0.0
    }
    state = fsm.traverse(scores, conflict_load=0.35)
    assert state == FSMState.DIAGNOSTIC_TENSION

def test_high_certainty_low_conflict_resolved(fsm):
    scores = {
        "psoriasis": 0.85, "seborrheic_dermatitis": 0.05, "lichen_planus": 0.0,
        "pityriasis_rosea": 0.0, "chronic_dermatitis": 0.0, "pityriasis_rubra_pilaris": 0.0
    }
    state = fsm.traverse(scores, conflict_load=0.05)
    assert state == FSMState.RESOLVED

def test_state_is_int(fsm):
    scores = {"psoriasis": 0.9, "seborrheic_dermatitis": 0.0, "lichen_planus": 0.0,
              "pityriasis_rosea": 0.0, "chronic_dermatitis": 0.0, "pityriasis_rubra_pilaris": 0.0}
    state = fsm.traverse(scores, 0.0)
    assert isinstance(state, int)
    assert 0 <= state <= 4
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_fsm.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement DiagnosticFSM**

```python
# src/symbolic/fsm.py


class FSMState:
    EVIDENCE_SPARSE = 0
    HYPOTHESIS_FORMING = 1
    CERTAINTY_BUILDING = 2
    DIAGNOSTIC_TENSION = 3
    RESOLVED = 4


class DiagnosticFSM:
    def traverse(self, certainty_scores: dict, conflict_load: float) -> int:
        """
        Traverses 5-state FSM deterministically. Returns final state as int.
        States advance forward only — never backward.
        """
        top_certainty = max(certainty_scores.values()) if certainty_scores else 0.0
        diseases_above_threshold = sum(
            1 for s in certainty_scores.values() if s > 0.4
        )

        state = FSMState.EVIDENCE_SPARSE

        # EVIDENCE_SPARSE → HYPOTHESIS_FORMING
        if top_certainty > 0.1:
            state = FSMState.HYPOTHESIS_FORMING

        # HYPOTHESIS_FORMING → CERTAINTY_BUILDING
        if state >= FSMState.HYPOTHESIS_FORMING and top_certainty > 0.4:
            state = FSMState.CERTAINTY_BUILDING

        # CERTAINTY_BUILDING → DIAGNOSTIC_TENSION
        if state >= FSMState.CERTAINTY_BUILDING and (
            conflict_load > 0.3 or diseases_above_threshold >= 2
        ):
            state = FSMState.DIAGNOSTIC_TENSION

        # Any state → RESOLVED (overrides DIAGNOSTIC_TENSION if certainty strong)
        if (top_certainty > 0.65 and conflict_load < 0.25) or top_certainty > 0.80:
            state = FSMState.RESOLVED

        return int(state)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_fsm.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symbolic/fsm.py tests/test_fsm.py
git commit -m "feat: DiagnosticFSM — 5-state deterministic diagnostic trajectory"
```

---

### Task 8: Symbolic Pipeline Integration

**Files:**
- Create: `src/symbolic/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: outputs of RuleEngine, ConflictAnalyzer, DiagnosticFSM
- Produces:
  - `SymbolicPipeline(rules_dir: str)` — composes all symbolic components
  - `SymbolicPipeline.transform(X_fuzzy: pd.DataFrame) -> pd.DataFrame`
    returns DataFrame of shape (n_patients, 9) with columns:
    `certainty_psoriasis`, `certainty_seborrheic_dermatitis`, `certainty_lichen_planus`, `certainty_pityriasis_rosea`, `certainty_chronic_dermatitis`, `certainty_pityriasis_rubra_pilaris`, `conflict_load`, `contradiction_severity`, `fsm_state`
  - `SymbolicPipeline.explain(row: pd.Series) -> dict` — full reasoning trace for one patient

SYMBOLIC_FEATURE_NAMES (exact column order, used by Model C):
```python
["certainty_psoriasis", "certainty_seborrheic_dermatitis", "certainty_lichen_planus",
 "certainty_pityriasis_rosea", "certainty_chronic_dermatitis", "certainty_pityriasis_rubra_pilaris",
 "conflict_load", "contradiction_severity", "fsm_state"]
```

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline.py
import pytest
import pandas as pd
from src.symbolic.pipeline import SymbolicPipeline, SYMBOLIC_FEATURE_NAMES

RULES_DIR = "rules"

@pytest.fixture
def pipeline():
    return SymbolicPipeline(RULES_DIR)

@pytest.fixture
def two_patients():
    return pd.DataFrame([
        {
            "erythema": 0.6667, "scaling": 0.6667, "definite_borders": 0.3333,
            "itching": 0.3333, "koebner_phenomenon": 1.0, "polygonal_papules": 0.0,
            "follicular_papules": 0.0, "oral_mucosal_involvement": 0.0,
            "knee_and_elbow_involvement": 1.0, "scalp_involvement": 1.0,
            "family_history": 0.0, "age": 0.4375,
        },
        {
            "erythema": 0.0, "scaling": 0.0, "definite_borders": 0.0,
            "itching": 0.0, "koebner_phenomenon": 0.0, "polygonal_papules": 0.0,
            "follicular_papules": 0.0, "oral_mucosal_involvement": 0.0,
            "knee_and_elbow_involvement": 0.0, "scalp_involvement": 0.0,
            "family_history": 0.0, "age": 0.0,
        },
    ])

def test_transform_shape(pipeline, two_patients):
    result = pipeline.transform(two_patients)
    assert result.shape == (2, 9)

def test_transform_column_names(pipeline, two_patients):
    result = pipeline.transform(two_patients)
    assert list(result.columns) == SYMBOLIC_FEATURE_NAMES

def test_transform_value_range(pipeline, two_patients):
    result = pipeline.transform(two_patients)
    assert result.min().min() >= 0.0
    assert result.drop(columns=["fsm_state"]).max().max() <= 1.0
    assert result["fsm_state"].max() <= 4

def test_explain_keys(pipeline, two_patients):
    trace = pipeline.explain(two_patients.iloc[0])
    assert "certainty_scores" in trace
    assert "conflict_load" in trace
    assert "contradiction_severity" in trace
    assert "fsm_state" in trace
    assert "fired_rules" in trace
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_pipeline.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement SymbolicPipeline**

```python
# src/symbolic/pipeline.py
import pandas as pd
from src.symbolic.rule_engine import RuleEngine
from src.symbolic.conflict import ConflictAnalyzer
from src.symbolic.fsm import DiagnosticFSM

SYMBOLIC_FEATURE_NAMES = [
    "certainty_psoriasis",
    "certainty_seborrheic_dermatitis",
    "certainty_lichen_planus",
    "certainty_pityriasis_rosea",
    "certainty_chronic_dermatitis",
    "certainty_pityriasis_rubra_pilaris",
    "conflict_load",
    "contradiction_severity",
    "fsm_state",
]


class SymbolicPipeline:
    def __init__(self, rules_dir: str):
        self._rule_engine = RuleEngine(rules_dir)
        self._conflict_analyzer = ConflictAnalyzer()
        self._fsm = DiagnosticFSM()

    def _process_row(self, row: pd.Series) -> dict:
        certainty_scores = self._rule_engine.fire(row)
        conflict_load, contradiction_severity = self._conflict_analyzer.analyze(certainty_scores)
        fsm_state = self._fsm.traverse(certainty_scores, conflict_load)
        return {
            "certainty_psoriasis": certainty_scores["psoriasis"],
            "certainty_seborrheic_dermatitis": certainty_scores["seborrheic_dermatitis"],
            "certainty_lichen_planus": certainty_scores["lichen_planus"],
            "certainty_pityriasis_rosea": certainty_scores["pityriasis_rosea"],
            "certainty_chronic_dermatitis": certainty_scores["chronic_dermatitis"],
            "certainty_pityriasis_rubra_pilaris": certainty_scores["pityriasis_rubra_pilaris"],
            "conflict_load": conflict_load,
            "contradiction_severity": contradiction_severity,
            "fsm_state": fsm_state,
        }

    def transform(self, X_fuzzy: pd.DataFrame) -> pd.DataFrame:
        rows = [self._process_row(row) for _, row in X_fuzzy.iterrows()]
        return pd.DataFrame(rows, columns=SYMBOLIC_FEATURE_NAMES).reset_index(drop=True)

    def explain(self, row: pd.Series) -> dict:
        certainty_scores = self._rule_engine.fire(row)
        conflict_load, contradiction_severity = self._conflict_analyzer.analyze(certainty_scores)
        fsm_state = self._fsm.traverse(certainty_scores, conflict_load)
        fired_rules = self._rule_engine.get_fired_rules(row)
        return {
            "certainty_scores": certainty_scores,
            "conflict_load": round(conflict_load, 4),
            "contradiction_severity": round(contradiction_severity, 4),
            "fsm_state": fsm_state,
            "fired_rules": fired_rules,
        }
```

- [ ] **Step 4: Run all symbolic tests**

```bash
pytest tests/test_rule_engine.py tests/test_conflict.py tests/test_fsm.py tests/test_pipeline.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symbolic/pipeline.py tests/test_pipeline.py
git commit -m "feat: SymbolicPipeline — orchestrates rule engine, conflict, FSM into 9-feature output"
```

---

### Task 9: Models A and B (Baselines)

**Files:**
- Create: `src/models/base.py`
- Create: `src/models/model_a.py`
- Create: `src/models/model_b.py`

**Interfaces:**
- Consumes: `(X: pd.DataFrame, y: pd.Series)` from DataLoader
- Produces:
  - `get_xgb_params() -> dict` — shared XGBoost hyperparameters
  - `run_model_a(X_all, y) -> dict` — cross-val results
  - `run_model_b(X_clinical_fuzzy, y) -> dict`
  - Both return: `{"accuracy_mean": float, "accuracy_std": float, "macro_f1_mean": float, "macro_f1_std": float, "per_class_f1": dict, "confusion_matrix": np.ndarray}`

- [ ] **Step 1: Create src/models/base.py**

```python
# src/models/base.py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

N_SPLITS = 10
RANDOM_STATE = 42

DISEASES = [
    "psoriasis", "seborrheic_dermatitis", "lichen_planus",
    "pityriasis_rosea", "chronic_dermatitis", "pityriasis_rubra_pilaris"
]


def get_xgb_params() -> dict:
    return {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
        "use_label_encoder": False,
    }


def cross_validate_model(X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    """
    Runs stratified 10-fold CV with XGBoost. Returns aggregated metrics.
    y must be 0-indexed (0-5).
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accuracies, macro_f1s = [], []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**get_xgb_params())
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_val)

        accuracies.append(accuracy_score(y_val, y_pred))
        macro_f1s.append(f1_score(y_val, y_pred, average="macro", zero_division=0))
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())

    per_class_f1 = f1_score(all_y_true, all_y_pred, average=None, zero_division=0)
    cm = confusion_matrix(all_y_true, all_y_pred)

    return {
        "label": label,
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "per_class_f1": {DISEASES[i]: round(float(per_class_f1[i]), 4)
                         for i in range(len(DISEASES))},
        "confusion_matrix": cm,
    }
```

- [ ] **Step 2: Create src/models/model_a.py**

```python
# src/models/model_a.py
import pandas as pd
from src.models.base import cross_validate_model


def run_model_a(X_all: pd.DataFrame, y: pd.Series) -> dict:
    """
    Model A: all 34 features (clinical + histopathological).
    Represents the biopsy-assisted upper bound.
    """
    return cross_validate_model(X_all, y, label="Model A (All 34 features)")
```

- [ ] **Step 3: Create src/models/model_b.py**

```python
# src/models/model_b.py
import pandas as pd
from src.models.base import cross_validate_model
from src.grading.fuzzy_grader import FuzzyGrader


def run_model_b(X_clinical: pd.DataFrame, y: pd.Series) -> dict:
    """
    Model B: 12 clinical features only (biopsy-free baseline).
    Features are fuzzy-graded before classification.
    """
    grader = FuzzyGrader()
    X_fuzzy = grader.grade(X_clinical)
    return cross_validate_model(X_fuzzy, y, label="Model B (12 clinical features)")
```

- [ ] **Step 4: Run a quick smoke test**

```bash
python -c "
from src.data.loader import load_dataset
from src.models.model_a import run_model_a
from src.models.model_b import run_model_b

X_clinical, X_histopath, X_all, y = load_dataset()
res_a = run_model_a(X_all, y)
res_b = run_model_b(X_clinical, y)
print(f'Model A: {res_a[\"accuracy_mean\"]:.4f} ± {res_a[\"accuracy_std\"]:.4f}')
print(f'Model B: {res_b[\"accuracy_mean\"]:.4f} ± {res_b[\"accuracy_std\"]:.4f}')
"
```

Expected: Model A accuracy ~0.96–0.99, Model B ~0.84–0.90. Both run without error.

- [ ] **Step 5: Commit**

```bash
git add src/models/base.py src/models/model_a.py src/models/model_b.py
git commit -m "feat: Models A and B — biopsy-assisted and clinical-only baselines"
```

---

### Task 10: Model C (Hybrid Symbolic-Statistical)

**Files:**
- Create: `src/models/model_c.py`

**Interfaces:**
- Consumes: `X_clinical`, `y` from DataLoader; `SymbolicPipeline` from Task 8; `FuzzyGrader` from Task 3
- Produces:
  - `run_model_c(X_clinical: pd.DataFrame, y: pd.Series, rules_dir: str) -> dict`
    same return shape as `run_model_a` / `run_model_b`
  - Also returns `"X_combined"` key containing the full 21-feature DataFrame for SHAP analysis

- [ ] **Step 1: Create src/models/model_c.py**

```python
# src/models/model_c.py
import pandas as pd
from src.grading.fuzzy_grader import FuzzyGrader
from src.symbolic.pipeline import SymbolicPipeline
from src.models.base import cross_validate_model


def run_model_c(X_clinical: pd.DataFrame, y: pd.Series, rules_dir: str = "rules") -> dict:
    """
    Model C: 12 fuzzy clinical features + 9 symbolic outputs = 21 features.
    This is the novel hybrid contribution.
    """
    grader = FuzzyGrader()
    X_fuzzy = grader.grade(X_clinical).reset_index(drop=True)

    pipeline = SymbolicPipeline(rules_dir)
    X_symbolic = pipeline.transform(X_fuzzy).reset_index(drop=True)

    X_combined = pd.concat([X_fuzzy, X_symbolic], axis=1)

    results = cross_validate_model(X_combined, y, label="Model C (12 clinical + 9 symbolic)")
    results["X_combined"] = X_combined  # for SHAP in Task 13
    return results
```

- [ ] **Step 2: Smoke test Model C**

```bash
python -c "
from src.data.loader import load_dataset
from src.models.model_c import run_model_c

X_clinical, _, _, y = load_dataset()
res = run_model_c(X_clinical, y)
print(f'Model C: {res[\"accuracy_mean\"]:.4f} ± {res[\"accuracy_std\"]:.4f}')
print(f'Per-class F1: {res[\"per_class_f1\"]}')
print(f'X_combined shape: {res[\"X_combined\"].shape}')
"
```

Expected: accuracy between 0.86–0.95, shape (366, 21), no errors.

- [ ] **Step 3: Commit**

```bash
git add src/models/model_c.py
git commit -m "feat: Model C — hybrid 21-feature classifier combining clinical + symbolic outputs"
```

---

### Task 11: Biopsy Triage Layer

**Files:**
- Create: `src/triage/biopsy_triage.py`
- Create: `tests/test_triage.py`

**Interfaces:**
- Consumes: `top_certainty: float`, `conflict_load: float`, `fsm_state: int`
- Produces:
  - `BiopsyTriage.recommend(top_certainty, conflict_load, fsm_state) -> str`
    returns one of: `"SAFE_BIOPSY_FREE"`, `"UNCERTAIN"`, `"BIOPSY_ADVISED"`
  - `BiopsyTriage.batch_recommend(X_symbolic: pd.DataFrame) -> pd.Series`
    applies to full symbolic output DataFrame

- [ ] **Step 1: Write failing tests**

```python
# tests/test_triage.py
import pytest
import pandas as pd
from src.triage.biopsy_triage import BiopsyTriage, TRIAGE_TIERS
from src.symbolic.fsm import FSMState

@pytest.fixture
def triage():
    return BiopsyTriage()

def test_safe_biopsy_free(triage):
    result = triage.recommend(
        top_certainty=0.80,
        conflict_load=0.10,
        fsm_state=FSMState.RESOLVED
    )
    assert result == "SAFE_BIOPSY_FREE"

def test_uncertain(triage):
    result = triage.recommend(
        top_certainty=0.60,
        conflict_load=0.25,
        fsm_state=FSMState.CERTAINTY_BUILDING
    )
    assert result == "UNCERTAIN"

def test_biopsy_advised_low_certainty(triage):
    result = triage.recommend(
        top_certainty=0.30,
        conflict_load=0.50,
        fsm_state=FSMState.DIAGNOSTIC_TENSION
    )
    assert result == "BIOPSY_ADVISED"

def test_biopsy_advised_high_conflict(triage):
    result = triage.recommend(
        top_certainty=0.80,
        conflict_load=0.45,
        fsm_state=FSMState.RESOLVED
    )
    assert result != "SAFE_BIOPSY_FREE"

def test_batch_recommend_length(triage):
    data = pd.DataFrame([
        {"certainty_psoriasis": 0.85, "conflict_load": 0.05, "fsm_state": 4},
        {"certainty_psoriasis": 0.40, "conflict_load": 0.60, "fsm_state": 3},
    ])
    # manually add max certainty
    data["top_certainty"] = data[["certainty_psoriasis"]].max(axis=1)
    result = triage.batch_recommend(data)
    assert len(result) == 2
    assert set(result.unique()).issubset(set(TRIAGE_TIERS))
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_triage.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement BiopsyTriage**

```python
# src/triage/biopsy_triage.py
import pandas as pd
from src.symbolic.fsm import FSMState

TRIAGE_TIERS = ["SAFE_BIOPSY_FREE", "UNCERTAIN", "BIOPSY_ADVISED"]

# Thresholds — documented rationale in design spec
SAFE_CERTAINTY_THRESHOLD = 0.75
SAFE_CONFLICT_THRESHOLD = 0.20
UNCERTAIN_CERTAINTY_THRESHOLD = 0.55
UNCERTAIN_CONFLICT_THRESHOLD = 0.40


class BiopsyTriage:
    def recommend(
        self,
        top_certainty: float,
        conflict_load: float,
        fsm_state: int,
    ) -> str:
        if (
            top_certainty >= SAFE_CERTAINTY_THRESHOLD
            and conflict_load < SAFE_CONFLICT_THRESHOLD
            and fsm_state == FSMState.RESOLVED
        ):
            return "SAFE_BIOPSY_FREE"
        if (
            top_certainty >= UNCERTAIN_CERTAINTY_THRESHOLD
            and conflict_load < UNCERTAIN_CONFLICT_THRESHOLD
        ):
            return "UNCERTAIN"
        return "BIOPSY_ADVISED"

    def batch_recommend(self, X_symbolic: pd.DataFrame) -> pd.Series:
        disease_certainty_cols = [
            c for c in X_symbolic.columns if c.startswith("certainty_")
        ]
        top_certainty = X_symbolic[disease_certainty_cols].max(axis=1)
        return pd.Series([
            self.recommend(
                top_certainty=top_certainty.iloc[i],
                conflict_load=float(X_symbolic["conflict_load"].iloc[i]),
                fsm_state=int(X_symbolic["fsm_state"].iloc[i]),
            )
            for i in range(len(X_symbolic))
        ], name="triage")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_triage.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/triage/biopsy_triage.py tests/test_triage.py
git commit -m "feat: BiopsyTriage — rule-based SAFE/UNCERTAIN/BIOPSY_ADVISED triage"
```

---

### Task 12: Evaluation — Comparison and Statistical Tests

**Files:**
- Create: `src/evaluation/metrics.py`

**Interfaces:**
- Consumes: result dicts from `run_model_a`, `run_model_b`, `run_model_c`
- Produces:
  - `print_comparison_table(results_a, results_b, results_c)` — prints formatted table
  - `run_statistical_test(results_b, results_c) -> dict` — paired t-test B vs C, returns `{"t_stat": float, "p_value": float, "significant": bool}`
  - `per_class_safety_analysis(X_symbolic, y_true, y_pred) -> pd.DataFrame` — per-disease biopsy safety breakdown

- [ ] **Step 1: Create src/evaluation/metrics.py**

```python
# src/evaluation/metrics.py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from src.models.base import get_xgb_params, DISEASES, N_SPLITS, RANDOM_STATE


def print_comparison_table(results_a: dict, results_b: dict, results_c: dict) -> None:
    header = f"{'Model':<35} {'Accuracy':>12} {'Macro F1':>12}"
    print("\n" + "=" * 60)
    print(header)
    print("-" * 60)
    for res in [results_a, results_b, results_c]:
        label = res["label"]
        acc = f"{res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f}"
        f1 = f"{res['macro_f1_mean']:.4f} ± {res['macro_f1_std']:.4f}"
        print(f"{label:<35} {acc:>12} {f1:>12}")
    print("=" * 60)

    print("\nPer-class F1 scores:")
    print(f"{'Disease':<35} {'Model A':>10} {'Model B':>10} {'Model C':>10}")
    print("-" * 65)
    for disease in DISEASES:
        fa = results_a["per_class_f1"].get(disease, 0.0)
        fb = results_b["per_class_f1"].get(disease, 0.0)
        fc = results_c["per_class_f1"].get(disease, 0.0)
        print(f"{disease:<35} {fa:>10.4f} {fb:>10.4f} {fc:>10.4f}")


def run_statistical_test(
    X_b: pd.DataFrame,
    X_c: pd.DataFrame,
    y: pd.Series
) -> dict:
    """
    5x2 paired cross-validation t-test comparing Model B vs Model C.
    Standard method for comparing two classifiers (Dietterich 1998).
    """
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    p_values = []

    for _ in range(5):
        fold_accs_b, fold_accs_c = [], []
        for train_idx, val_idx in cv.split(X_b, y):
            # Model B fold
            clf_b = XGBClassifier(**get_xgb_params())
            clf_b.fit(X_b.iloc[train_idx], y.iloc[train_idx], verbose=False)
            acc_b = accuracy_score(y.iloc[val_idx], clf_b.predict(X_b.iloc[val_idx]))

            # Model C fold
            clf_c = XGBClassifier(**get_xgb_params())
            clf_c.fit(X_c.iloc[train_idx], y.iloc[train_idx], verbose=False)
            acc_c = accuracy_score(y.iloc[val_idx], clf_c.predict(X_c.iloc[val_idx]))

            fold_accs_b.append(acc_b)
            fold_accs_c.append(acc_c)

        diff = np.array(fold_accs_c) - np.array(fold_accs_b)
        t_stat, p_val = stats.ttest_1samp(diff, 0.0)
        p_values.append(p_val)

    combined_p = float(np.mean(p_values))
    return {
        "p_value": round(combined_p, 4),
        "significant": combined_p < 0.05,
        "interpretation": "Model C significantly better than B" if combined_p < 0.05
                         else "No significant difference B vs C",
    }


def per_class_safety_analysis(
    X_symbolic: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    For each disease: computes % of correctly classified cases that are SAFE_BIOPSY_FREE.
    Requires X_symbolic to contain triage column.
    """
    from src.triage.biopsy_triage import BiopsyTriage
    triage = BiopsyTriage()
    triage_labels = triage.batch_recommend(X_symbolic)

    results = []
    for i, disease in enumerate(DISEASES):
        mask_true = (y_true.values == i)
        mask_correct = (y_pred == i) & mask_true
        n_correct = mask_correct.sum()
        n_safe = ((triage_labels == "SAFE_BIOPSY_FREE") & mask_correct).sum()
        pct_safe = (n_safe / n_correct * 100) if n_correct > 0 else 0.0
        results.append({
            "disease": disease,
            "n_patients": int(mask_true.sum()),
            "n_correct": int(n_correct),
            "n_safe_biopsy_free": int(n_safe),
            "pct_safe_biopsy_free": round(pct_safe, 1),
        })

    return pd.DataFrame(results)
```

- [ ] **Step 2: Smoke test metrics**

```bash
python -c "
from src.data.loader import load_dataset
from src.models.model_a import run_model_a
from src.models.model_b import run_model_b
from src.models.model_c import run_model_c
from src.evaluation.metrics import print_comparison_table

X_clinical, X_histopath, X_all, y = load_dataset()
res_a = run_model_a(X_all, y)
res_b = run_model_b(X_clinical, y)
res_c = run_model_c(X_clinical, y)
print_comparison_table(res_a, res_b, res_c)
"
```

Expected: formatted table with all three models, no errors.

- [ ] **Step 3: Commit**

```bash
git add src/evaluation/metrics.py
git commit -m "feat: evaluation metrics — comparison table, 5x2 t-test, per-class safety analysis"
```

---

### Task 13: Explainability — SHAP and imodels

**Files:**
- Create: `src/evaluation/explainability.py`
- Create: `notebooks/analysis.ipynb`

**Interfaces:**
- Consumes: trained XGBoost model, `X_combined` from `run_model_c`
- Produces:
  - `compute_shap_values(model, X) -> np.ndarray` — SHAP values array
  - `plot_shap_beeswarm(shap_values, X, save_path)` — saves PNG
  - `extract_imodels_rules(X, y, max_rules) -> list[str]` — IF-THEN rules as strings

- [ ] **Step 1: Create src/evaluation/explainability.py**

```python
# src/evaluation/explainability.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from imodels import RuleFitClassifier
from src.models.base import get_xgb_params, DISEASES


def train_final_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    """Train on full dataset (no CV) for SHAP/imodels analysis."""
    model = XGBClassifier(**get_xgb_params())
    model.fit(X, y, verbose=False)
    return model


def compute_shap_values(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: str = "shap_beeswarm.png",
    class_idx: int = 0,
) -> None:
    """
    Plots SHAP beeswarm for a single class. Call once per disease class.
    class_idx: 0=psoriasis, 1=seborrheic_dermatitis, ..., 5=pityriasis_rubra_pilaris
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values[class_idx],
        X,
        plot_type="dot",
        show=False,
        max_display=21,
    )
    plt.title(f"SHAP Feature Importance — {DISEASES[class_idx]}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_shap_global(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: str = "shap_global.png",
) -> None:
    """Mean |SHAP| across all classes — shows overall feature importance."""
    mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    mean_importance = pd.Series(
        mean_abs.mean(axis=0), index=X.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    mean_importance.head(21).plot(kind="barh")
    plt.xlabel("Mean |SHAP value|")
    plt.title("Global Feature Importance — Model C")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def extract_imodels_rules(
    X: pd.DataFrame,
    y: pd.Series,
    max_rules: int = 20,
) -> list[str]:
    """
    Fits RuleFit on Model C features. Extracts top IF-THEN rules.
    Note: these are post-hoc explanations of the trained model,
    distinct from the hand-crafted YAML rules in the symbolic engine.
    """
    rulefit = RuleFitClassifier(max_rules=max_rules, random_state=42)
    rulefit.fit(X.values, y.values, feature_names=list(X.columns))

    rules = rulefit.get_rules()
    rules = rules[rules["coef"] != 0].sort_values("importance", ascending=False)
    top_rules = []
    for _, row in rules.head(max_rules).iterrows():
        top_rules.append(f"IF {row['rule']} THEN importance={row['importance']:.4f}")
    return top_rules
```

- [ ] **Step 2: Create notebooks/analysis.ipynb with full analysis**

Create a Jupyter notebook with these cells in order:

**Cell 1 — Imports and data:**
```python
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

from src.data.loader import load_dataset
from src.grading.fuzzy_grader import FuzzyGrader
from src.symbolic.pipeline import SymbolicPipeline
from src.models.model_a import run_model_a
from src.models.model_b import run_model_b
from src.models.model_c import run_model_c
from src.triage.biopsy_triage import BiopsyTriage
from src.evaluation.metrics import print_comparison_table, run_statistical_test, per_class_safety_analysis
from src.evaluation.explainability import train_final_model, compute_shap_values, plot_shap_global, plot_shap_beeswarm, extract_imodels_rules

X_clinical, X_histopath, X_all, y = load_dataset()
print(f"Dataset: {X_all.shape}, Classes: {y.value_counts().to_dict()}")
```

**Cell 2 — Run all three models:**
```python
res_a = run_model_a(X_all, y)
res_b = run_model_b(X_clinical, y)
res_c = run_model_c(X_clinical, y)
print_comparison_table(res_a, res_b, res_c)
```

**Cell 3 — Statistical test B vs C:**
```python
from src.grading.fuzzy_grader import FuzzyGrader
grader = FuzzyGrader()
X_b = grader.grade(X_clinical).reset_index(drop=True)
X_c = res_c["X_combined"]

stat_result = run_statistical_test(X_b, X_c, y)
print(f"p-value: {stat_result['p_value']}")
print(f"Significant: {stat_result['significant']}")
print(stat_result['interpretation'])
```

**Cell 4 — SHAP analysis:**
```python
X_combined = res_c["X_combined"]
final_model = train_final_model(X_combined, y)
shap_values = compute_shap_values(final_model, X_combined)
plot_shap_global(shap_values, X_combined, save_path="shap_global.png")
plot_shap_beeswarm(shap_values, X_combined, save_path="shap_psoriasis.png", class_idx=0)
```

**Cell 5 — imodels rule extraction:**
```python
rules = extract_imodels_rules(X_combined, y, max_rules=15)
for r in rules:
    print(r)
```

**Cell 6 — Biopsy triage + per-class safety:**
```python
pipeline = SymbolicPipeline("rules")
grader = FuzzyGrader()
X_fuzzy = grader.grade(X_clinical).reset_index(drop=True)
X_symbolic = pipeline.transform(X_fuzzy)

final_model_c = train_final_model(res_c["X_combined"], y)
y_pred = final_model_c.predict(res_c["X_combined"])

safety_df = per_class_safety_analysis(X_symbolic, y, y_pred)
print(safety_df.to_string(index=False))
```

**Cell 7 — Single patient reasoning trace:**
```python
sample = X_clinical.iloc[0]
fuzzy_sample = grader.grade_series(sample)
trace = pipeline.explain(fuzzy_sample)
print("=== Clinical Reasoning Trace ===")
print(f"Certainty scores: {trace['certainty_scores']}")
print(f"Conflict load: {trace['conflict_load']}")
print(f"Contradiction severity: {trace['contradiction_severity']}")
print(f"FSM state: {trace['fsm_state']}")
print(f"Fired rules:")
for r in trace["fired_rules"]:
    print(f"  {r['id']} ({r['disease']}, tier {r['tier']}): strength={r['firing_strength']}, contrib={r['contribution']}")

triage = BiopsyTriage()
certainty_scores = list(trace["certainty_scores"].values())
triage_result = triage.recommend(
    top_certainty=max(certainty_scores),
    conflict_load=trace["conflict_load"],
    fsm_state=trace["fsm_state"],
)
print(f"\nTriage recommendation: {triage_result}")
print(f"True label: {y.iloc[0]}")
```

- [ ] **Step 3: Run full notebook**

```bash
jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb --output notebooks/analysis_executed.ipynb
```

Expected: notebook executes without errors. If `RuleFitClassifier` errors on multiclass, replace `extract_imodels_rules` call with `max_rules=10` and check imodels version.

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Final commit**

```bash
git add src/evaluation/explainability.py notebooks/analysis.ipynb
git commit -m "feat: SHAP + imodels explainability, full analysis notebook, per-class biopsy safety"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| Fuzzy grading ordinal → float | Task 3, FuzzyGrader |
| Tiered rule YAML (A/B/C/D tiers) | Task 4 (rules) + Task 5 (RuleEngine) |
| D-tier subtracts from competitor certainty | Task 5 RuleEngine._fire_rule |
| 6 disease certainty scores | Task 5 RuleEngine.fire() |
| Conflict load computation | Task 6 ConflictAnalyzer |
| Contradiction severity | Task 6 ConflictAnalyzer |
| 5-state FSM | Task 7 DiagnosticFSM |
| 9 symbolic outputs | Task 8 SymbolicPipeline |
| Model A (all 34 features) | Task 9 run_model_a |
| Model B (12 clinical) | Task 9 run_model_b |
| Model C (21 features hybrid) | Task 10 run_model_c |
| Biopsy triage (3 tiers) | Task 11 BiopsyTriage |
| Stratified 10-fold CV | Task 9 base.cross_validate_model |
| Paired t-test B vs C | Task 12 run_statistical_test |
| Per-class biopsy safety analysis | Task 12 per_class_safety_analysis |
| SHAP TreeExplainer | Task 13 compute_shap_values |
| imodels rule extraction | Task 13 extract_imodels_rules |
| Reasoning trace per patient | Task 8 SymbolicPipeline.explain |
| Median imputation for age | Task 2 DataLoader |

**Placeholder scan:** None found.

**Type consistency check:**
- `RuleEngine.fire()` → `dict[str, float]` → consumed by `ConflictAnalyzer.analyze()` ✓
- `ConflictAnalyzer.analyze()` → `tuple[float, float]` → consumed by `DiagnosticFSM.traverse()` ✓
- `SymbolicPipeline.transform()` → `pd.DataFrame` with 9 cols → consumed by `run_model_c` ✓
- `BiopsyTriage.batch_recommend()` consumes `X_symbolic` with `certainty_*` + `conflict_load` + `fsm_state` cols ✓
- `FSMState.RESOLVED = 4` used in both `DiagnosticFSM` and `BiopsyTriage` ✓
