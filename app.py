"""
HSCIS-ESD — Biopsy-Free Differential Diagnosis
Streamlit inference interface. Trains on full UCI dataset at startup,
then predicts in real-time from 12 observable clinical features.
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import defaultdict
from catboost import CatBoostClassifier

from src.data.loader import load_dataset
from src.grading.fuzzy_grader import FuzzyGrader
from src.grading.feature_engineer import FeatureEngineer
from src.symbolic.pipeline import SymbolicPipeline
from src.symbolic.rule_engine import RuleEngine
from src.models.base import get_catboost_params_c, DISEASES
from src.triage.biopsy_triage import BiopsyTriage

st.set_page_config(
    page_title="HSCIS-ESD",
    page_icon="🔬",
    layout="wide",
)

# ── Design system ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background-color: #070c14 !important;
    color: #e2e8f0 !important;
}

/* ── Hide Streamlit chrome ─────────────────────────── */
#MainMenu, footer,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ── Layout ────────────────────────────────────────── */
.main .block-container {
    padding: 0 2.5rem 3rem !important;
    max-width: 1160px !important;
}

/* ── Scrollbar ─────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.10); border-radius: 3px; }

/* ── Paragraph / caption ───────────────────────────── */
p, span, div, label {
    font-family: 'Inter', system-ui, sans-serif !important;
}
[data-testid="stCaptionContainer"] p {
    font-size: 0.72rem !important;
    color: #334155 !important;
}
hr { border-color: rgba(255,255,255,0.05) !important; margin: 1rem 0 !important; }

/* ── Form labels ───────────────────────────────────── */
[data-testid="stSlider"] label p,
[data-testid="stRadio"] > label p,
[data-testid="stNumberInput"] label p {
    font-size: 0.79rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    letter-spacing: 0 !important;
}

/* ── Slider ────────────────────────────────────────── */
[data-testid="stSlider"] { padding-bottom: 6px !important; }
[data-testid="stSlider"] > div > div > div > div {
    background: #2563eb !important;
}
[data-testid="stSlider"] > div > div > div > div > div {
    background: #3b82f6 !important;
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.18) !important;
}

/* ── Radio ─────────────────────────────────────────── */
[data-testid="stRadio"] > div {
    flex-direction: row !important;
    gap: 6px !important;
    flex-wrap: wrap !important;
}
[data-testid="stRadio"] > div > label {
    background: #0a1422 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 6px !important;
    padding: 4px 12px !important;
    font-size: 0.77rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    cursor: pointer !important;
    transition: border-color 0.12s, color 0.12s, background 0.12s !important;
}
[data-testid="stRadio"] > div > label:has(input:checked) {
    background: rgba(37,99,235,0.10) !important;
    border-color: rgba(37,99,235,0.35) !important;
    color: #93c5fd !important;
}

/* ── Number input ──────────────────────────────────── */
[data-testid="stNumberInput"] input {
    background: #0a1422 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 7px !important;
    color: #e2e8f0 !important;
    font-size: 0.86rem !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: rgba(37,99,235,0.45) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.10) !important;
    outline: none !important;
}

/* ── Bordered containers ───────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: #0b1626 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 11px !important;
    padding: 16px 18px !important;
}

/* ── Button ────────────────────────────────────────── */
[data-testid="stButton"] > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    letter-spacing: 0.005em !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
}
[data-testid="stButton"] > button[kind="primary"] {
    background: #2563eb !important;
    border: none !important;
    color: #fff !important;
    padding: 0.65rem 1.5rem !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #1d4ed8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(37,99,235,0.28) !important;
}
[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(0) !important;
    box-shadow: none !important;
}

/* ── Metrics ───────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #0b1626 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 9px !important;
    padding: 13px 16px !important;
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.60rem !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.10em !important;
    color: #334155 !important; font-family: 'Inter', sans-serif !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 1.05rem !important; font-weight: 700 !important;
    color: #e2e8f0 !important; font-family: 'Inter', sans-serif !important;
    font-variant-numeric: tabular-nums !important;
}
[data-testid="stMetricDelta"] { display: none !important; }

/* ── Progress bar ──────────────────────────────────── */
[data-testid="stProgress"] > div {
    background: #0f1e32 !important;
    border-radius: 4px !important;
    height: 5px !important;
    overflow: hidden !important;
}
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #1d4ed8, #60a5fa) !important;
    border-radius: 4px !important;
    height: 5px !important;
}
[data-testid="stProgress"] p { display: none !important; }

/* ── Expander ──────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0b1626 !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 8px !important;
    margin-top: 6px !important;
}
[data-testid="stExpander"] summary p {
    font-size: 0.79rem !important; font-weight: 500 !important;
    color: #334155 !important; font-family: 'Inter', sans-serif !important;
}
[data-testid="stExpander"] > div > div > div {
    padding: 8px 4px !important;
}

/* ── ─────────────── Custom components ────────────── */

.nav {
    background: rgba(7,12,20,0.96);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin: 0 -2.5rem 28px;
    padding: 16px 2.5rem;
    position: sticky; top: 0; z-index: 999;
    display: flex; align-items: center; justify-content: space-between;
}
.nav-left { display: flex; align-items: center; gap: 10px; }
.nav-title { font-size: 0.93rem; font-weight: 700; color: #e2e8f0; letter-spacing: -0.01em; }
.nav-badge {
    font-size: 0.58rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.10em;
    color: #60a5fa; background: rgba(37,99,235,0.10); border: 1px solid rgba(37,99,235,0.22);
    padding: 2px 8px; border-radius: 4px;
}
.nav-right { display: flex; gap: 32px; }
.nav-stat { text-align: right; }
.nav-stat-val { font-size: 0.84rem; font-weight: 700; color: #e2e8f0; font-variant-numeric: tabular-nums; }
.nav-stat-lbl { font-size: 0.58rem; text-transform: uppercase; letter-spacing: 0.09em; color: #334155; }

.form-sec {
    font-size: 0.60rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.13em; color: #334155; margin: 20px 0 8px;
}

.dx-card {
    background: #0a1828;
    border: 1px solid rgba(37,99,235,0.20);
    border-top: 2px solid #2563eb;
    border-radius: 12px; padding: 22px 24px; margin-bottom: 14px;
}
.dx-eye {
    font-size: 0.58rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.15em; color: #2563eb; margin-bottom: 7px;
}
.dx-name {
    font-size: 1.75rem; font-weight: 800; color: #f1f5f9;
    letter-spacing: -0.025em; line-height: 1.12; margin-bottom: 4px;
}
.dx-conf { font-size: 0.75rem; color: #334155; }
.dx-conf span { color: #60a5fa; font-weight: 600; font-variant-numeric: tabular-nums; }

.conf-row {
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;
}
.conf-lbl { font-size: 0.60rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.10em; color: #334155; }
.conf-pct { font-size: 0.78rem; font-weight: 700; color: #60a5fa; font-variant-numeric: tabular-nums; }

.tr-card { border-radius: 9px; padding: 12px 16px; margin: 10px 0 18px; }
.tr-title { font-size: 0.86rem; font-weight: 700; color: #f1f5f9; margin-bottom: 2px; }
.tr-sub { font-size: 0.73rem; color: #64748b; }
.tr-safe    { background: rgba(16,185,129,0.06); border: 1px solid rgba(16,185,129,0.18); border-left: 3px solid #10b981; }
.tr-unsure  { background: rgba(245,158,11,0.06); border: 1px solid rgba(245,158,11,0.18); border-left: 3px solid #f59e0b; }
.tr-biopsy  { background: rgba(239,68,68,0.06); border: 1px solid rgba(239,68,68,0.18); border-left: 3px solid #ef4444; }

.ch-lbl {
    font-size: 0.60rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.10em; color: #334155; margin: 20px 0 4px;
}

.ev-lbl {
    font-size: 0.58rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: #1e293b; margin: 18px 0 7px;
}
.ev {
    background: #0b1626; border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px; padding: 10px 14px; margin: 5px 0;
    font-size: 0.81rem; line-height: 1.6; color: #94a3b8;
}
.ev strong { color: #cbd5e1; font-weight: 600; }
.ev .str { float: right; font-size: 0.67rem; color: #1e293b; margin-top: 2px; font-variant-numeric: tabular-nums; }
.ev-x {
    background: rgba(99,102,241,0.04); border: 1px solid rgba(99,102,241,0.10);
    border-radius: 8px; padding: 10px 14px; margin: 5px 0;
    font-size: 0.81rem; line-height: 1.6; color: #64748b;
}
.ev-x strong { color: #94a3b8; font-weight: 600; }

.about {
    background: #0b1626; border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px; padding: 26px 28px;
}
.about-h { font-size: 1.0rem; font-weight: 700; color: #e2e8f0; letter-spacing: -0.01em; margin-bottom: 10px; }
.about-p { font-size: 0.82rem; color: #475569; line-height: 1.8; margin-bottom: 22px; }
.st-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.81rem;
}
.st-row:last-child { border-bottom: none; padding-bottom: 0; }
.st-k { color: #475569; }
.st-v { color: #e2e8f0; font-weight: 600; font-variant-numeric: tabular-nums; }
.about-legend { margin-top: 18px; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.04); }
.about-legend-row { font-size: 0.75rem; color: #334155; line-height: 2.1; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DISEASE_LABELS = {
    "psoriasis":               "Psoriasis",
    "seborrheic_dermatitis":   "Seborrheic Dermatitis",
    "lichen_planus":           "Lichen Planus",
    "pityriasis_rosea":        "Pityriasis Rosea",
    "chronic_dermatitis":      "Chronic Dermatitis",
    "pityriasis_rubra_pilaris":"Pityriasis Rubra Pilaris",
}
FEATURE_LABELS = {
    "erythema":                 "skin redness (erythema)",
    "scaling":                  "scaling",
    "definite_borders":         "well-defined lesion borders",
    "itching":                  "itching (pruritus)",
    "koebner_phenomenon":       "Koebner phenomenon",
    "polygonal_papules":        "polygonal papules",
    "follicular_papules":       "follicular papules",
    "oral_mucosal_involvement": "oral mucosal involvement",
    "knee_elbow_involvement":   "knee / elbow involvement",
    "scalp_involvement":        "scalp involvement",
    "family_history":           "positive family history",
    "age":                      "patient age",
}
TIER_LABEL = {"A": "Highly specific sign for", "B": "Commonly seen in", "C": "Occasionally seen in"}
TIER_ICON  = {"A": "🔴", "B": "🟠", "C": "🟡"}
TRIAGE_CFG = {
    "SAFE_BIOPSY_FREE": ("tr-safe",   "🟢", "Safe — Biopsy Not Required",       "High certainty, low diagnostic conflict"),
    "UNCERTAIN":        ("tr-unsure", "🟡", "Uncertain — Use Clinical Judgement","Moderate evidence; specialist referral may be warranted"),
    "BIOPSY_ADVISED":   ("tr-biopsy", "🔴", "Biopsy Advised",                   "Ambiguous evidence; histopathology recommended"),
}

# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    X_clinical, _, _, y = load_dataset()
    grader   = FuzzyGrader()
    X_fuzzy  = grader.grade(X_clinical).reset_index(drop=True)
    engineer = FeatureEngineer()
    X_eng    = engineer.engineer(X_fuzzy)
    pipeline = SymbolicPipeline("rules")
    X_sym    = pipeline.transform(X_fuzzy).reset_index(drop=True)
    X_train  = pd.concat([X_fuzzy, X_eng, X_sym], axis=1)
    model    = CatBoostClassifier(**get_catboost_params_c())
    model.fit(X_train, y.reset_index(drop=True))
    return model, grader, engineer, pipeline, RuleEngine("rules")


def predict(model, grader, engineer, sym_pipeline, raw, rule_engine):
    X_raw     = pd.DataFrame([raw])
    X_fuzzy   = grader.grade(X_raw).reset_index(drop=True)
    X_eng     = engineer.engineer(X_fuzzy)
    X_sym     = sym_pipeline.transform(X_fuzzy).reset_index(drop=True)
    X_all     = pd.concat([X_fuzzy, X_eng, X_sym], axis=1)
    proba     = model.predict_proba(X_all)[0]
    pred_idx  = int(np.argmax(proba))
    pred_dis  = DISEASES[pred_idx]
    confidence= float(proba[pred_idx])
    cert_cols = [c for c in X_sym.columns if c.startswith("certainty_")]
    top_cert  = float(X_sym[cert_cols].iloc[0].max())
    conflict  = float(X_sym["conflict_load"].iloc[0]) if "conflict_load" in X_sym.columns else 0.0
    fsm_val   = int(X_sym["fsm_state"].iloc[0])       if "fsm_state"    in X_sym.columns else 0
    rec       = BiopsyTriage().recommend(top_certainty=top_cert, conflict_load=conflict, fsm_state=fsm_val)
    FSM_NAMES = ["Evidence Sparse","Hypothesis Forming","Building Evidence","Diagnostic Tension","Resolved"]
    return {
        "disease":     pred_dis,
        "confidence":  confidence,
        "proba":       {DISEASES[i]: float(proba[i]) for i in range(len(DISEASES))},
        "recommendation": rec,
        "sym_cert":    {d: float(X_sym[f"certainty_{d}"].iloc[0]) for d in DISEASES if f"certainty_{d}" in X_sym.columns},
        "fsm":         FSM_NAMES[min(fsm_val, 4)],
        "conflict":    conflict,
        "fired":       rule_engine.get_fired_rules(X_fuzzy.iloc[0]),
    }


def ev_strength(c):
    return "Strong" if c >= 0.80 else "Moderate" if c >= 0.50 else "Weak"


def altair_bar(df, x_col, bar_color, height=175):
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3, color=bar_color)
        .encode(
            x=alt.X(f"{x_col}:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%", labelColor="#334155",
                                  gridColor="rgba(255,255,255,0.03)",
                                  domainOpacity=0, tickOpacity=0,
                                  labelFontFamily="Inter", labelFontSize=10)),
            y=alt.Y("Disease:N", sort="-x", title=None,
                    axis=alt.Axis(labelColor="#64748b", domainOpacity=0,
                                  tickOpacity=0, labelFontFamily="Inter", labelFontSize=11)),
            tooltip=["Disease:N", alt.Tooltip(f"{x_col}:Q", format=".1%", title=x_col)],
        )
        .properties(height=height, background="transparent")
        .configure_view(strokeWidth=0, fill="transparent")
    )


# ── Navigation bar ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav">
  <div class="nav-left">
    <span style="font-size:1.05rem;">🔬</span>
    <span class="nav-title">HSCIS-ESD</span>
    <span class="nav-badge">Biopsy-Free</span>
  </div>
  <div class="nav-right">
    <div class="nav-stat">
      <div class="nav-stat-val">88.79%</div>
      <div class="nav-stat-lbl">Accuracy</div>
    </div>
    <div class="nav-stat">
      <div class="nav-stat-val">0.8850</div>
      <div class="nav-stat-lbl">Macro F1</div>
    </div>
    <div class="nav-stat">
      <div class="nav-stat-val">p = 0.0176</div>
      <div class="nav-stat-lbl">McNemar</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

model, grader, engineer, sym_pipeline, rule_engine = load_model()

left, right = st.columns([1, 1.55], gap="large")

# ── Input panel ────────────────────────────────────────────────────────────────
with left:
    st.markdown('<p class="form-sec">Severity Signs</p>', unsafe_allow_html=True)
    with st.container(border=True):
        erythema         = st.slider("Erythema",        0, 3, 1)
        scaling          = st.slider("Scaling",         0, 3, 1)
        definite_borders = st.slider("Definite borders",0, 3, 1)
        itching          = st.slider("Itching",         0, 3, 1)
        st.caption("0 = absent  ·  1 = mild  ·  2 = moderate  ·  3 = severe")

    st.markdown('<p class="form-sec">Clinical Signs</p>', unsafe_allow_html=True)
    with st.container(border=True):
        koebner            = st.radio("Koebner phenomenon",       [0,1], format_func=lambda x:"Present" if x else "Absent", horizontal=True)
        polygonal_papules  = st.radio("Polygonal papules",        [0,1], format_func=lambda x:"Present" if x else "Absent", horizontal=True)
        follicular_papules = st.radio("Follicular papules",       [0,1], format_func=lambda x:"Present" if x else "Absent", horizontal=True)
        oral_mucosal       = st.radio("Oral mucosal involvement", [0,1], format_func=lambda x:"Present" if x else "Absent", horizontal=True)
        knee_elbow         = st.radio("Knee / elbow involvement", [0,1], format_func=lambda x:"Present" if x else "Absent", horizontal=True)
        scalp              = st.radio("Scalp involvement",        [0,1], format_func=lambda x:"Present" if x else "Absent", horizontal=True)
        family_history     = st.radio("Family history",           [0,1], format_func=lambda x:"Positive" if x else "Negative", horizontal=True)

    st.markdown('<p class="form-sec">Patient</p>', unsafe_allow_html=True)
    age = st.number_input("Age (years)", min_value=1, max_value=100, value=35)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Run Diagnosis →", type="primary", use_container_width=True)

# ── Output panel ───────────────────────────────────────────────────────────────
with right:
    if run_btn:
        raw = {
            "erythema": erythema, "scaling": scaling,
            "definite_borders": definite_borders, "itching": itching,
            "koebner_phenomenon": koebner, "polygonal_papules": polygonal_papules,
            "follicular_papules": follicular_papules, "oral_mucosal_involvement": oral_mucosal,
            "knee_elbow_involvement": knee_elbow, "scalp_involvement": scalp,
            "family_history": family_history, "age": age,
        }
        with st.spinner("Analysing clinical profile…"):
            res = predict(model, grader, engineer, sym_pipeline, raw, rule_engine)

        dlabel  = DISEASE_LABELS.get(res["disease"], res["disease"])
        conf    = res["confidence"]
        rec     = res["recommendation"]
        tr_cls, tr_ico, tr_title, tr_sub = TRIAGE_CFG.get(rec, ("tr-unsure","⚪",rec,""))

        # ── Diagnosis ──────────────────────────────────────
        st.markdown(f"""
<div class="dx-card">
  <p class="dx-eye">Primary Diagnosis</p>
  <p class="dx-name">{dlabel}</p>
  <p class="dx-conf">Classifier confidence &nbsp;<span>{conf*100:.1f}%</span></p>
</div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="conf-row"><span class="conf-lbl">Confidence</span><span class="conf-pct">{conf*100:.1f}%</span></div>', unsafe_allow_html=True)
        st.progress(conf)

        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence",  f"{conf*100:.1f}%")
        m2.metric("Diagnostic State", res["fsm"].split()[0])
        m3.metric("Conflict Load", f"{res['conflict']:.3f}")

        # ── Triage ─────────────────────────────────────────
        st.markdown(f"""
<div class="tr-card {tr_cls}">
  <p class="tr-title">{tr_ico} {tr_title}</p>
  <p class="tr-sub">{tr_sub}</p>
</div>""", unsafe_allow_html=True)

        # ── Probability chart ───────────────────────────────
        st.markdown('<p class="ch-lbl">Classifier Probabilities</p>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            "Disease": [DISEASE_LABELS[d] for d in DISEASES],
            "Probability": [res["proba"][d] for d in DISEASES],
        })
        st.altair_chart(altair_bar(prob_df, "Probability", "#2563eb"), use_container_width=True)

        # ── Symbolic certainty chart ───────────────────────
        if res["sym_cert"]:
            st.markdown('<p class="ch-lbl">Expert Rule Certainty</p>', unsafe_allow_html=True)
            cert_df = pd.DataFrame({
                "Disease": [DISEASE_LABELS[d] for d in DISEASES if d in res["sym_cert"]],
                "Certainty": [res["sym_cert"][d] for d in DISEASES if d in res["sym_cert"]],
            })
            st.altair_chart(altair_bar(cert_df, "Certainty", "#7c3aed"), use_container_width=True)

        # ── Clinical reasoning ──────────────────────────────
        st.markdown('<p class="ch-lbl" style="margin-top:22px;">Clinical Reasoning</p>', unsafe_allow_html=True)
        fired = res.get("fired", [])

        if fired:
            pred_dis   = res["disease"]
            pred_label = DISEASE_LABELS.get(pred_dis, pred_dis)
            primary    = [r for r in fired if r["disease"] == pred_dis and r["tier"] != "D"]

            if primary:
                tg = defaultdict(lambda: {"features": [], "mc": 0.0})
                for r in sorted(primary, key=lambda x: -x["contribution"]):
                    t = r["tier"]
                    for f in r.get("conditions", []):
                        lbl = FEATURE_LABELS.get(f, f)
                        if lbl not in tg[t]["features"]:
                            tg[t]["features"].append(lbl)
                    if r["contribution"] > tg[t]["mc"]:
                        tg[t]["mc"] = r["contribution"]

                for tk in ["A", "B", "C"]:
                    if tk not in tg: continue
                    icon  = TIER_ICON.get(tk, "")
                    tier  = TIER_LABEL.get(tk, tk)
                    signs = ", ".join(tg[tk]["features"])
                    s     = ev_strength(tg[tk]["mc"])
                    st.markdown(f"""<div class="ev">{icon} <strong>{tier} {pred_label}</strong><span class="str">{s}</span><br>{signs}</div>""", unsafe_allow_html=True)
            else:
                st.caption(f"No specific rules fired for {pred_label} — prediction driven by statistical pattern.")

            penalising = [r for r in fired if r["tier"] == "D"]
            if penalising:
                st.markdown('<p class="ev-lbl" style="margin-top:14px;">Signs that argue against other diagnoses</p>', unsafe_allow_html=True)
                sign_map = {}
                for r in penalising:
                    s = ", ".join(FEATURE_LABELS.get(f, f) for f in r.get("conditions", []))
                    d = DISEASE_LABELS.get(r["disease"], r["disease"])
                    sign_map.setdefault(s, [])
                    if d not in sign_map[s]: sign_map[s].append(d)
                for sign, diseases in sign_map.items():
                    st.markdown(f"""<div class="ev-x">⬇️ <strong>{sign}</strong> — not typical for {", ".join(diseases)}</div>""", unsafe_allow_html=True)

            other = [r for r in fired if r["disease"] != pred_dis and r["tier"] != "D"]
            if other:
                with st.expander("Full evidence trail — all diseases"):
                    dtg = {}
                    for r in other:
                        key = (DISEASE_LABELS.get(r["disease"], r["disease"]), r["tier"])
                        dtg.setdefault(key, {"features": [], "mc": 0.0})
                        for f in r.get("conditions", []):
                            lbl = FEATURE_LABELS.get(f, f)
                            if lbl not in dtg[key]["features"]: dtg[key]["features"].append(lbl)
                        if r["contribution"] > dtg[key]["mc"]: dtg[key]["mc"] = r["contribution"]
                    for (dis, tk), g in sorted(dtg.items(), key=lambda x: (-x[1]["mc"], x[0][0])):
                        icon  = TIER_ICON.get(tk, "")
                        tier  = TIER_LABEL.get(tk, tk)
                        signs = ", ".join(g["features"])
                        s     = ev_strength(g["mc"])
                        st.markdown(f"{icon} **{tier} {dis}** — *{signs}* ({s})")
        else:
            st.caption("No expert rules fired — prediction driven entirely by statistical classifier.")

    else:
        # ── About card (pre-diagnosis) ──────────────────────
        st.markdown("""
<div class="about">
  <p class="about-h">Erythemato-Squamous Disease Diagnosis</p>
  <p class="about-p">
    HSCIS-ESD differentiates six clinically similar skin diseases using only
    12 observable clinical features — no biopsy required. The system combines
    fuzzy symptom grading, 41 expert-encoded diagnostic rules across four
    evidence tiers, and a CatBoost classifier trained on 366 patients from
    the UCI Dermatology Dataset.
  </p>
  <div class="st-row"><span class="st-k">Accuracy</span><span class="st-v">88.79% ±3.34%</span></div>
  <div class="st-row"><span class="st-k">Macro F1</span><span class="st-v">0.8850</span></div>
  <div class="st-row"><span class="st-k">vs. clinical baseline</span><span class="st-v">+3.84 pp &nbsp;(McNemar p = 0.0176)</span></div>
  <div class="st-row"><span class="st-k">Expert rules</span><span class="st-v">41 rules · 4 evidence tiers</span></div>
  <div class="st-row"><span class="st-k">Variance reduction</span><span class="st-v">45% (±6.01% → ±3.34%)</span></div>
  <div class="st-row"><span class="st-k">Training patients</span><span class="st-v">366 · UCI Dermatology Dataset</span></div>
  <div class="about-legend">
    <div class="about-legend-row">🟢 &nbsp;<strong style="color:#94a3b8;">Safe</strong> &nbsp;— high certainty, low conflict</div>
    <div class="about-legend-row">🟡 &nbsp;<strong style="color:#94a3b8;">Uncertain</strong> &nbsp;— moderate evidence, clinical judgement required</div>
    <div class="about-legend-row">🔴 &nbsp;<strong style="color:#94a3b8;">Biopsy advised</strong> &nbsp;— ambiguous evidence, histopathology recommended</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.caption("UCI Dermatology Dataset · 366 patients · 6 ESD classes · CC BY 4.0 · Ridanshi Agarwal")
