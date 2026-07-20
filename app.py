"""
HSCIS-ESD — Biopsy-Free Differential Diagnosis
Streamlit inference interface. Trains on full UCI dataset at startup,
then predicts in real-time from 12 observable clinical features.
"""
import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
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

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Page background */
    .stApp { background-color: #0f1117; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* Header */
    .app-header {
        padding: 8px 0 4px 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 24px;
    }
    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.78rem;
        color: rgba(255,255,255,0.45);
        margin: 2px 0 0 0;
        letter-spacing: 0.03em;
    }

    /* Section headers */
    .section-title {
        font-size: 0.70rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        color: rgba(255,255,255,0.40);
        margin: 18px 0 8px 0;
    }

    /* Diagnosis result card */
    .diagnosis-card {
        background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(34,197,94,0.04));
        border: 1px solid rgba(34,197,94,0.25);
        border-radius: 12px;
        padding: 20px 24px;
        margin: 0 0 16px 0;
    }
    .diagnosis-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        color: rgba(34,197,94,0.8);
        margin: 0 0 6px 0;
    }
    .diagnosis-name {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
    }

    /* Triage badges */
    .triage-safe {
        background: rgba(34,197,94,0.12);
        border: 1px solid rgba(34,197,94,0.30);
        border-left: 4px solid #22c55e;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 12px 0;
        color: #ffffff;
        font-size: 0.92rem;
    }
    .triage-uncertain {
        background: rgba(250,204,21,0.10);
        border: 1px solid rgba(250,204,21,0.28);
        border-left: 4px solid #facc15;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 12px 0;
        color: #ffffff;
        font-size: 0.92rem;
    }
    .triage-biopsy {
        background: rgba(239,68,68,0.10);
        border: 1px solid rgba(239,68,68,0.28);
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 12px 0;
        color: #ffffff;
        font-size: 0.92rem;
    }
    .triage-title {
        font-weight: 700;
        font-size: 0.95rem;
        margin: 0 0 2px 0;
    }
    .triage-subtitle {
        font-size: 0.78rem;
        opacity: 0.65;
        margin: 0;
    }

    /* Evidence items */
    .evidence-block {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.88rem;
        line-height: 1.5;
        color: rgba(255,255,255,0.88);
    }
    .evidence-block-exclusion {
        background: rgba(99,102,241,0.06);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.88rem;
        color: rgba(255,255,255,0.75);
    }
    .evidence-heading {
        font-size: 0.70rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255,255,255,0.35);
        margin: 16px 0 6px 0;
    }

    /* About card */
    .about-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 20px 24px;
        margin: 0;
    }

    /* Metric overrides */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* Input section container */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px !important;
    }

    /* Radio buttons horizontal */
    div[role="radiogroup"] {
        gap: 8px;
    }

    /* Confidence bar label */
    .conf-label {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.40);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 12px 0 4px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DISEASE_LABELS = {
    "psoriasis": "Psoriasis",
    "seborrheic_dermatitis": "Seborrheic Dermatitis",
    "lichen_planus": "Lichen Planus",
    "pityriasis_rosea": "Pityriasis Rosea",
    "chronic_dermatitis": "Chronic Dermatitis",
    "pityriasis_rubra_pilaris": "Pityriasis Rubra Pilaris",
}

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

TIER_LABEL = {"A": "Highly specific sign for", "B": "Commonly seen in", "C": "Occasionally seen in"}
TIER_ICON  = {"A": "🔴", "B": "🟠", "C": "🟡"}

TRIAGE_MESSAGES = {
    "SAFE_BIOPSY_FREE": ("🟢 Safe — Biopsy Not Required", "High certainty, low diagnostic conflict"),
    "UNCERTAIN":        ("🟡 Uncertain — Use Clinical Judgement", "Moderate evidence; consider specialist referral"),
    "BIOPSY_ADVISED":  ("🔴 Biopsy Advised", "Ambiguous evidence; histopathology recommended"),
}
TRIAGE_CLASS = {
    "SAFE_BIOPSY_FREE": "triage-safe",
    "UNCERTAIN":        "triage-uncertain",
    "BIOPSY_ADVISED":  "triage-biopsy",
}

# ── Model loading ──────────────────────────────────────────────────────────────
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
    fsm_state_names = ["Evidence Sparse", "Hypothesis Forming", "Building Evidence", "Diagnostic Tension", "Resolved"]
    fsm_val = int(X_sym["fsm_state"].iloc[0]) if "fsm_state" in X_sym.columns else 0
    fsm_name = fsm_state_names[min(fsm_val, 4)]
    conflict = float(X_sym["conflict_load"].iloc[0]) if "conflict_load" in X_sym.columns else 0.0

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


def evidence_label(contribution):
    if contribution >= 0.80:
        return "Strong evidence"
    elif contribution >= 0.50:
        return "Moderate evidence"
    return "Weak evidence"


# ── Page header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <p class="app-title">🔬 HSCIS-ESD</p>
    <p class="app-subtitle">BIOPSY-FREE DIFFERENTIAL DIAGNOSIS &nbsp;·&nbsp; 88.79% ACCURACY &nbsp;·&nbsp; McNEMAR p=0.0176</p>
</div>
""", unsafe_allow_html=True)

model, grader, engineer, sym_pipeline, rule_engine = load_model()

col1, col2 = st.columns([1, 1.3], gap="large")

# ── Left column: inputs ────────────────────────────────────────────────────────
with col1:
    st.markdown('<p class="section-title">Severity Signs</p>', unsafe_allow_html=True)
    with st.container(border=True):
        erythema         = st.slider("Erythema (skin redness)",  0, 3, 1)
        scaling          = st.slider("Scaling (skin shedding)",  0, 3, 1)
        definite_borders = st.slider("Definite borders",         0, 3, 1)
        itching          = st.slider("Itching (pruritus)",       0, 3, 1)
        st.caption("0 = absent · 1 = mild · 2 = moderate · 3 = severe")

    st.markdown('<p class="section-title">Clinical Signs</p>', unsafe_allow_html=True)
    with st.container(border=True):
        koebner = st.radio(
            "Koebner phenomenon", [0, 1],
            format_func=lambda x: "Present" if x else "Absent",
            horizontal=True
        )
        polygonal_papules = st.radio(
            "Polygonal papules", [0, 1],
            format_func=lambda x: "Present" if x else "Absent",
            horizontal=True
        )
        follicular_papules = st.radio(
            "Follicular papules", [0, 1],
            format_func=lambda x: "Present" if x else "Absent",
            horizontal=True
        )
        oral_mucosal = st.radio(
            "Oral mucosal involvement", [0, 1],
            format_func=lambda x: "Present" if x else "Absent",
            horizontal=True
        )
        knee_elbow = st.radio(
            "Knee / elbow involvement", [0, 1],
            format_func=lambda x: "Present" if x else "Absent",
            horizontal=True
        )
        scalp = st.radio(
            "Scalp involvement", [0, 1],
            format_func=lambda x: "Present" if x else "Absent",
            horizontal=True
        )
        family_history = st.radio(
            "Family history", [0, 1],
            format_func=lambda x: "Positive" if x else "Negative",
            horizontal=True
        )

    st.markdown('<p class="section-title">Patient</p>', unsafe_allow_html=True)
    age = st.number_input("Age (years)", min_value=1, max_value=100, value=35)

    predict_btn = st.button("Run Diagnosis →", type="primary", use_container_width=True)

# ── Right column: output ───────────────────────────────────────────────────────
with col2:
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

        rec = result["recommendation"]
        disease_label = DISEASE_LABELS.get(result["disease"], result["disease"])
        confidence_pct = result["confidence"] * 100

        # Primary diagnosis card
        st.markdown(f"""
<div class="diagnosis-card">
    <p class="diagnosis-label">Primary Diagnosis</p>
    <p class="diagnosis-name">{disease_label}</p>
</div>
""", unsafe_allow_html=True)

        # Confidence bar
        st.markdown('<p class="conf-label">Classifier Confidence</p>', unsafe_allow_html=True)
        st.progress(result["confidence"], text=f"{confidence_pct:.1f}%")

        # 3 metrics in a row
        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence", f"{confidence_pct:.1f}%")
        m2.metric("Diagnostic State", result["fsm_state"].split()[0])
        m3.metric("Conflict Load", f"{result['conflict_load']:.3f}")

        # Triage recommendation
        triage_title, triage_sub = TRIAGE_MESSAGES.get(rec, (rec, ""))
        triage_cls = TRIAGE_CLASS.get(rec, "triage-uncertain")
        st.markdown(f"""
<div class="{triage_cls}">
    <p class="triage-title">{triage_title}</p>
    <p class="triage-subtitle">{triage_sub}</p>
</div>
""", unsafe_allow_html=True)

        # Probability chart
        st.markdown('<p class="section-title" style="margin-top:20px;">Classifier Probabilities</p>', unsafe_allow_html=True)
        proba_df = pd.DataFrame({
            "Disease": [DISEASE_LABELS[d] for d in DISEASES],
            "Probability": [result["proba"][d] for d in DISEASES],
        }).sort_values("Probability", ascending=False)
        st.bar_chart(proba_df.set_index("Disease"), horizontal=True, color="#22c55e")

        # Symbolic certainty
        if result["sym_certainties"]:
            st.markdown('<p class="section-title">Expert Rule Certainty Scores</p>', unsafe_allow_html=True)
            cert_df = pd.DataFrame({
                "Disease": [DISEASE_LABELS[d] for d in DISEASES if d in result["sym_certainties"]],
                "Certainty": [result["sym_certainties"][d] for d in DISEASES if d in result["sym_certainties"]],
            }).sort_values("Certainty", ascending=False)
            st.bar_chart(cert_df.set_index("Disease"), horizontal=True, color="#818cf8")

        # Reasoning trace
        st.markdown('<p class="section-title" style="margin-top:20px;">Clinical Reasoning</p>', unsafe_allow_html=True)
        fired = result.get("fired_rules", [])

        if fired:
            pred_disease = result["disease"]
            pred_label = DISEASE_LABELS.get(pred_disease, pred_disease)

            # Primary: rules for predicted disease grouped by tier
            primary = [r for r in fired if r["disease"] == pred_disease and r["tier"] != "D"]
            primary_sorted = sorted(primary, key=lambda x: -x["contribution"])

            if primary_sorted:
                tier_groups = defaultdict(lambda: {"features": [], "max_contribution": 0.0})
                for r in primary_sorted:
                    t = r["tier"]
                    for f in r.get("conditions", []):
                        lbl = FEATURE_LABELS.get(f, f)
                        if lbl not in tier_groups[t]["features"]:
                            tier_groups[t]["features"].append(lbl)
                    if r["contribution"] > tier_groups[t]["max_contribution"]:
                        tier_groups[t]["max_contribution"] = r["contribution"]

                for tier_key in ["A", "B", "C"]:
                    if tier_key not in tier_groups:
                        continue
                    icon = TIER_ICON.get(tier_key, "")
                    tier = TIER_LABEL.get(tier_key, tier_key)
                    signs = ", ".join(tier_groups[tier_key]["features"])
                    lbl = evidence_label(tier_groups[tier_key]["max_contribution"])
                    st.markdown(f"""
<div class="evidence-block">
    {icon} <strong>{tier} {pred_label}</strong><br>
    <span style="opacity:0.7;">{signs}</span>
    <span style="float:right;font-size:0.75rem;opacity:0.5;">{lbl}</span>
</div>
""", unsafe_allow_html=True)
            else:
                st.caption(f"No specific expert rules fired for {pred_label} — diagnosis driven by statistical pattern.")

            # Exclusion rules
            penalising = [r for r in fired if r["tier"] == "D"]
            if penalising:
                st.markdown('<p class="evidence-heading">Signs that argue against other diagnoses</p>', unsafe_allow_html=True)
                sign_to_diseases = {}
                for r in penalising:
                    signs = ", ".join(FEATURE_LABELS.get(f, f) for f in r.get("conditions", []))
                    disease = DISEASE_LABELS.get(r["disease"], r["disease"])
                    if signs not in sign_to_diseases:
                        sign_to_diseases[signs] = []
                    if disease not in sign_to_diseases[signs]:
                        sign_to_diseases[signs].append(disease)
                for sign, diseases in sign_to_diseases.items():
                    st.markdown(f"""
<div class="evidence-block-exclusion">
    ⬇️ <strong>{sign}</strong> — not typical for {', '.join(diseases)}
</div>
""", unsafe_allow_html=True)

            # Full evidence trail collapsed
            other = [r for r in fired if r["disease"] != pred_disease and r["tier"] != "D"]
            if other:
                with st.expander("Full evidence trail (all diseases)"):
                    dt_groups = {}
                    for r in other:
                        key = (DISEASE_LABELS.get(r["disease"], r["disease"]), r["tier"])
                        if key not in dt_groups:
                            dt_groups[key] = {"features": [], "max_contribution": 0.0}
                        for f in r.get("conditions", []):
                            lbl = FEATURE_LABELS.get(f, f)
                            if lbl not in dt_groups[key]["features"]:
                                dt_groups[key]["features"].append(lbl)
                        if r["contribution"] > dt_groups[key]["max_contribution"]:
                            dt_groups[key]["max_contribution"] = r["contribution"]
                    for (disease, tier_key), g in sorted(dt_groups.items(), key=lambda x: (-x[1]["max_contribution"], x[0][0])):
                        icon = TIER_ICON.get(tier_key, "")
                        tier = TIER_LABEL.get(tier_key, tier_key)
                        signs = ", ".join(g["features"])
                        lbl = evidence_label(g["max_contribution"])
                        st.markdown(f"{icon} **{tier} {disease}** — *{signs}* ({lbl})")
        else:
            st.caption("No expert rules fired — prediction driven entirely by statistical classifier.")

    else:
        st.markdown("""
<div class="about-card">
    <p style="font-size:1rem;font-weight:600;margin:0 0 12px 0;">About this system</p>
    <p style="font-size:0.88rem;opacity:0.7;line-height:1.7;margin:0 0 16px 0;">
        HSCIS-ESD diagnoses six erythemato-squamous skin diseases using only
        12 observable clinical features — no biopsy required. It combines
        fuzzy grading, 41 expert-encoded diagnostic rules, and a CatBoost
        statistical classifier trained on 366 patients.
    </p>
    <table style="width:100%;font-size:0.82rem;border-collapse:collapse;">
        <tr>
            <td style="padding:5px 0;opacity:0.5;">Accuracy</td>
            <td style="padding:5px 0;font-weight:600;">88.79% ±3.34%</td>
        </tr>
        <tr>
            <td style="padding:5px 0;opacity:0.5;">Macro F1</td>
            <td style="padding:5px 0;font-weight:600;">0.8850</td>
        </tr>
        <tr>
            <td style="padding:5px 0;opacity:0.5;">Statistical test</td>
            <td style="padding:5px 0;font-weight:600;">McNemar p = 0.0176</td>
        </tr>
        <tr>
            <td style="padding:5px 0;opacity:0.5;">Expert rules</td>
            <td style="padding:5px 0;font-weight:600;">41 across 4 evidence tiers</td>
        </tr>
    </table>
    <hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:16px 0;">
    <p style="font-size:0.78rem;opacity:0.4;margin:0;">
        🟢 Safe — high certainty, low conflict &nbsp;·&nbsp;
        🟡 Uncertain — use clinical judgement &nbsp;·&nbsp;
        🔴 Biopsy advised — ambiguous evidence
    </p>
</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.caption("UCI Dermatology Dataset · 366 patients · 6 ESD classes · CC BY 4.0 · Ridanshi Agarwal")
