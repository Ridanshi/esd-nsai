"""
HSCIS-ESD — Biopsy-Free Differential Diagnosis
Streamlit inference interface. Trains on full UCI dataset at startup,
then predicts in real-time from 12 observable clinical features.
"""
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

from src.data.loader import load_dataset
from src.grading.fuzzy_grader import FuzzyGrader
from src.grading.feature_engineer import FeatureEngineer
from src.symbolic.pipeline import SymbolicPipeline
from src.symbolic.rule_engine import RuleEngine
from src.models.base import get_catboost_params_c, DISEASES
from src.triage.biopsy_triage import BiopsyTriage

st.set_page_config(
    page_title="HSCIS-ESD Diagnosis",
    page_icon="🔬",
    layout="wide",
)

DISEASE_LABELS = {
    "psoriasis": "Psoriasis",
    "seborrheic_dermatitis": "Seborrheic Dermatitis",
    "lichen_planus": "Lichen Planus",
    "pityriasis_rosea": "Pityriasis Rosea",
    "chronic_dermatitis": "Chronic Dermatitis",
    "pityriasis_rubra_pilaris": "Pityriasis Rubra Pilaris",
}

TRIAGE_COLOR = {
    "SAFE_BIOPSY_FREE": "🟢",
    "UNCERTAIN": "🟡",
    "BIOPSY_ADVISED": "🔴",
}


@st.cache_resource(show_spinner="Training model on UCI dataset...")
def load_model():
    X_clinical, _, _, y = load_dataset()
    grader = FuzzyGrader()
    X_fuzzy = grader.grade(X_clinical).reset_index(drop=True)
    engineer = FeatureEngineer()
    X_eng = engineer.engineer(X_fuzzy)
    pipeline = SymbolicPipeline("rules")
    X_sym = pipeline.transform(X_fuzzy).reset_index(drop=True)
    X_train = pd.concat([X_fuzzy, X_eng, X_sym], axis=1)
    y_train = y.reset_index(drop=True)

    model = CatBoostClassifier(**get_catboost_params_c())
    model.fit(X_train, y_train)
    rule_engine = RuleEngine("rules")
    return model, grader, engineer, pipeline, rule_engine


def predict(model, grader, engineer, sym_pipeline, raw_input: dict, rule_engine=None):
    X_raw = pd.DataFrame([raw_input])
    X_fuzzy = grader.grade(X_raw).reset_index(drop=True)
    X_eng = engineer.engineer(X_fuzzy)
    X_sym = sym_pipeline.transform(X_fuzzy).reset_index(drop=True)
    X_combined = pd.concat([X_fuzzy, X_eng, X_sym], axis=1)

    proba = model.predict_proba(X_combined)[0]
    pred_idx = int(np.argmax(proba))
    pred_disease = DISEASES[pred_idx]
    confidence = float(proba[pred_idx])

    triage = BiopsyTriage()
    cert_cols = [c for c in X_sym.columns if c.startswith("certainty_")]
    top_certainty = float(X_sym[cert_cols].iloc[0].max())
    recommendation = triage.recommend(
        top_certainty=top_certainty,
        conflict_load=float(X_sym["conflict_load"].iloc[0]),
        fsm_state=int(X_sym["fsm_state"].iloc[0]),
    )

    sym_certainties = {
        d: float(X_sym[f"certainty_{d}"].iloc[0]) for d in DISEASES
        if f"certainty_{d}" in X_sym.columns
    }
    fsm_state_names = ["EVIDENCE_SPARSE", "HYPOTHESIS_FORMING", "BUILDING_EVIDENCE", "DIAGNOSTIC_TENSION", "RESOLVED"]
    fsm_val = int(X_sym["fsm_state"].iloc[0]) if "fsm_state" in X_sym.columns else 0
    fsm_name = fsm_state_names[min(fsm_val, 4)]
    conflict = float(X_sym["conflict_load"].iloc[0]) if "conflict_load" in X_sym.columns else 0.0

    # Fired rules for reasoning trace
    fired_rules = []
    if rule_engine is not None:
        fired_rules = rule_engine.get_fired_rules(X_fuzzy.iloc[0])

    return {
        "disease": pred_disease,
        "confidence": confidence,
        "proba": {DISEASES[i]: float(proba[i]) for i in range(len(DISEASES))},
        "recommendation": recommendation,
        "sym_certainties": sym_certainties,
        "fsm_state": fsm_name,
        "conflict_load": conflict,
        "fired_rules": fired_rules,
    }


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🔬 HSCIS-ESD — Biopsy-Free Differential Diagnosis")
st.caption("Hybrid Symbolic Clinical Inference System · 88.79% accuracy · McNemar p=0.0176")
st.divider()

model, grader, engineer, sym_pipeline, rule_engine = load_model()

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("Clinical Features")
    st.caption("Ordinal: 0 = absent, 3 = maximum severity")

    erythema           = st.slider("Erythema (skin redness)",        0, 3, 1)
    scaling            = st.slider("Scaling (skin shedding)",        0, 3, 1)
    definite_borders   = st.slider("Definite borders",               0, 3, 1)
    itching            = st.slider("Itching (pruritus)",             0, 3, 1)

    st.caption("Binary: 0 = absent, 1 = present")
    koebner            = st.selectbox("Koebner phenomenon",          [0, 1], format_func=lambda x: "Present" if x else "Absent")
    polygonal_papules  = st.selectbox("Polygonal papules",           [0, 1], format_func=lambda x: "Present" if x else "Absent")
    follicular_papules = st.selectbox("Follicular papules",          [0, 1], format_func=lambda x: "Present" if x else "Absent")
    oral_mucosal       = st.selectbox("Oral mucosal involvement",    [0, 1], format_func=lambda x: "Present" if x else "Absent")
    knee_elbow         = st.selectbox("Knee/elbow involvement",      [0, 1], format_func=lambda x: "Present" if x else "Absent")
    scalp              = st.selectbox("Scalp involvement",           [0, 1], format_func=lambda x: "Present" if x else "Absent")
    family_history     = st.selectbox("Family history",              [0, 1], format_func=lambda x: "Positive" if x else "Negative")
    age                = st.number_input("Patient age (years)",      min_value=1, max_value=100, value=35)

    predict_btn = st.button("🔍 Run Diagnosis", type="primary", use_container_width=True)

with col2:
    st.subheader("Diagnostic Output")

    if predict_btn:
        raw = {
            "erythema": erythema,
            "scaling": scaling,
            "definite_borders": definite_borders,
            "itching": itching,
            "koebner_phenomenon": koebner,
            "polygonal_papules": polygonal_papules,
            "follicular_papules": follicular_papules,
            "oral_mucosal_involvement": oral_mucosal,
            "knee_elbow_involvement": knee_elbow,
            "scalp_involvement": scalp,
            "family_history": family_history,
            "age": age,
        }

        with st.spinner("Analysing..."):
            result = predict(model, grader, engineer, sym_pipeline, raw, rule_engine)

        # Primary diagnosis
        rec = result["recommendation"]
        icon = TRIAGE_COLOR.get(rec, "⚪")
        disease_label = DISEASE_LABELS.get(result["disease"], result["disease"])

        st.success(f"**Predicted Diagnosis: {disease_label}**")
        st.metric("Classifier Confidence", f"{result['confidence']*100:.1f}%")

        st.divider()

        # Biopsy triage
        st.markdown(f"**Biopsy Recommendation:** {icon} `{rec}`")
        st.caption(f"Symbolic FSM state: `{result['fsm_state']}` · Conflict load: `{result['conflict_load']:.3f}`")

        st.divider()

        # Classifier probabilities
        st.markdown("**Classifier Probabilities (all 6 diseases)**")
        proba_df = pd.DataFrame({
            "Disease": [DISEASE_LABELS[d] for d in DISEASES],
            "Probability": [result["proba"][d] for d in DISEASES],
        }).sort_values("Probability", ascending=False)
        st.bar_chart(proba_df.set_index("Disease"), horizontal=True)

        # Symbolic certainty scores
        if result["sym_certainties"]:
            st.divider()
            st.markdown("**Symbolic Certainty Scores (expert rules)**")
            cert_df = pd.DataFrame({
                "Disease": [DISEASE_LABELS[d] for d in DISEASES if d in result["sym_certainties"]],
                "Certainty": [result["sym_certainties"][d] for d in DISEASES if d in result["sym_certainties"]],
            }).sort_values("Certainty", ascending=False)
            st.bar_chart(cert_df.set_index("Disease"), horizontal=True)

        # Reasoning trace — why this diagnosis
        st.divider()
        st.markdown("**Why this diagnosis? (Clinical reasoning)**")
        fired = result.get("fired_rules", [])

        FEATURE_LABELS = {
            "erythema": "skin redness (erythema)",
            "scaling": "scaling",
            "definite_borders": "well-defined lesion borders",
            "itching": "itching (pruritus)",
            "koebner_phenomenon": "Koebner phenomenon",
            "polygonal_papules": "polygonal papules",
            "follicular_papules": "follicular papules",
            "oral_mucosal_involvement": "oral mucosal involvement",
            "knee_elbow_involvement": "knee/elbow involvement",
            "scalp_involvement": "scalp involvement",
            "family_history": "positive family history",
            "age": "patient age",
        }
        TIER_LABEL = {"A": "Pathognomonic", "B": "Supportive", "C": "Auxiliary", "D": "Penalising"}
        TIER_ICON  = {"A": "🔴", "B": "🟠", "C": "🟡", "D": "⬇️"}

        if fired:
            supporting = [r for r in fired if r["tier"] != "D"]
            penalising = [r for r in fired if r["tier"] == "D"]

            def evidence_label(contribution):
                if contribution >= 0.80:
                    return "Strong evidence"
                elif contribution >= 0.50:
                    return "Moderate evidence"
                else:
                    return "Weak evidence"

            if supporting:
                st.markdown("*Signs that **support** a diagnosis:*")
                for r in sorted(supporting, key=lambda x: -x["contribution"]):
                    icon = TIER_ICON.get(r["tier"], "")
                    tier = TIER_LABEL.get(r["tier"], r["tier"])
                    disease = DISEASE_LABELS.get(r["disease"], r["disease"])
                    signs = ", ".join(
                        FEATURE_LABELS.get(f, f) for f in r.get("conditions", [])
                    )
                    label = evidence_label(r["contribution"])
                    st.markdown(
                        f"{icon} **{label} for {disease}** ({tier}) — "
                        f"triggered by: *{signs}*"
                    )

            if penalising:
                st.markdown("*Signs that **rule out** other diagnoses:*")
                for r in sorted(penalising, key=lambda x: -x["firing_strength"]):
                    disease = DISEASE_LABELS.get(r["disease"], r["disease"])
                    signs = ", ".join(
                        FEATURE_LABELS.get(f, f) for f in r.get("conditions", [])
                    )
                    st.markdown(
                        f"⬇️ **{disease} less likely** — "
                        f"*{signs}* not consistent with this disease"
                    )
        else:
            st.caption("No expert rules fired — prediction driven entirely by statistical classifier.")

    else:
        st.info("Set clinical feature values on the left, then click **Run Diagnosis**.")
        st.markdown("""
**About this system:**
- Uses only 12 observable clinical features (no biopsy required)
- Combines fuzzy grading, 41 expert-encoded rules, and CatBoost classification
- Accuracy: **88.79%** · Macro F1: **0.8850**
- Statistically validated (McNemar p=0.0176)

**Biopsy recommendation guide:**
- 🟢 SAFE_BIOPSY_FREE — high certainty, low conflict
- 🟡 UNCERTAIN — moderate evidence, proceed with clinical judgement
- 🔴 BIOPSY_ADVISED — ambiguous evidence, histopathology recommended
""")

st.divider()
st.caption("UCI Dermatology Dataset · 366 patients · 6 ESD classes · CC BY 4.0 · Ridanshi Agarwal")
