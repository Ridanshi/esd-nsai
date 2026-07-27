"""
HSCIS-ESD — Biopsy-Free Differential Diagnosis
Streamlit inference interface. Trains on full UCI dataset at startup,
then predicts in real-time from 12 observable clinical features.
"""
import streamlit as st
import streamlit.components.v1 as components
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
    layout="wide",
)

# ── Design system ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}
html, body { overflow-x: hidden !important; }

/* ── Hide Streamlit chrome ─────────────────────────── */
#MainMenu, footer,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ── Suppress rerun dim-fade ────────────────────────── */
[data-stale="true"] { opacity: 1 !important; transition: none !important; }

/* ── Layout ────────────────────────────────────────── */
[data-testid="stMainBlockContainer"] {
    padding: 0 2.5rem 3rem !important;
    max-width: 1600px !important;
    margin: 0 auto !important;
}
[data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
    padding-top: 0 !important;
}
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] > [data-testid="stMarkdownContainer"] > style) {
    display: none !important;
}

/* ── Scrollbar ─────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(15,23,42,0.14); border-radius: 3px; }

/* ── Paragraph / caption ───────────────────────────── */
p, span, div, label {
    font-family: 'Inter', system-ui, sans-serif !important;
}
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
}
[data-testid="stCaptionContainer"] p {
    font-size: 0.72rem !important;
    color: #94a3b8 !important;
    text-align: center !important;
}
hr { border-color: #e2e8f0 !important; margin: 1rem 0 !important; }

/* ── Form labels ───────────────────────────────────── */
[data-testid="stSlider"] label p,
[data-testid="stRadio"] > label p,
[data-testid="stNumberInput"] label p {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #1e293b !important;
    letter-spacing: 0 !important;
}

/* ── Slider ────────────────────────────────────────── */
[data-testid="stSlider"] { padding-bottom: 0 !important; }
[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
    background: #e2e8f0 !important;
    height: 4px !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: #2563eb !important;
    width: 15px !important;
    height: 15px !important;
    border: 2px solid #ffffff !important;
    box-shadow: 0 0 0 1px rgba(37,99,235,0.25), 0 1px 3px rgba(15,23,42,0.15) !important;
}
[data-testid="stSlider"] [data-testid="stSliderThumbValue"],
[data-testid="stSlider"] [data-testid="stSliderTickBar"] {
    display: none !important;
}
.tick-row { display: flex; justify-content: space-between; margin: -10px 0 14px; }
.tick-off { font-size: 0.76rem; font-weight: 600; color: #94a3b8; }
.tick-on  { font-size: 0.76rem; font-weight: 800; color: #2563eb; }

/* ── Radio ─────────────────────────────────────────── */
[data-testid="stRadio"] > div[role="radiogroup"] {
    flex-direction: row !important;
    gap: 6px !important;
    flex-wrap: wrap !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"] {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    padding: 4px 12px !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: border-color 0.12s, color 0.12s, background 0.12s !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-of-type,
[data-testid="stRadio"] label[data-baseweb="radio"] input {
    display: none !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"] p {
    font-size: 0.79rem !important;
    font-weight: 500 !important;
    color: #475569 !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    background: #eff6ff !important;
    border-color: #2563eb !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
    color: #2563eb !important;
}

/* ── Number input ──────────────────────────────────── */
[data-testid="stNumberInputContainer"] [data-baseweb="input"],
[data-testid="stNumberInputContainer"] [data-baseweb="base-input"] {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 7px !important;
}
[data-testid="stNumberInputField"] {
    background: #ffffff !important;
    color: #0f172a !important;
    font-size: 0.86rem !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stNumberInputContainer"] [data-baseweb="input"]:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.10) !important;
}
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {
    background: #ffffff !important;
    color: #64748b !important;
}

/* ── Bordered containers ───────────────────────────── */
[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
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
    padding: 0.7rem 1.5rem !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #1d4ed8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(37,99,235,0.22) !important;
}
[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(0) !important;
    box-shadow: none !important;
}

/* ── Metrics ───────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 9px !important;
    padding: 13px 16px !important;
    text-align: center !important;
}
[data-testid="stMetricLabel"] { justify-content: center !important; }
[data-testid="stMetricLabel"] > div {
    font-size: 0.60rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.10em !important;
    color: #94a3b8 !important; font-family: 'Inter', sans-serif !important;
}
[data-testid="stMetricValue"] { justify-content: center !important; }
[data-testid="stMetricValue"] > div {
    font-size: 1.05rem !important; font-weight: 700 !important;
    color: #0f172a !important; font-family: 'Inter', sans-serif !important;
    font-variant-numeric: tabular-nums !important;
}
[data-testid="stMetricDelta"] { display: none !important; }

/* ── Expander ──────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    margin-top: 6px !important;
}
[data-testid="stExpander"] summary p {
    font-size: 0.79rem !important; font-weight: 500 !important;
    color: #475569 !important; font-family: 'Inter', sans-serif !important;
}
[data-testid="stExpander"] > div > div > div {
    padding: 8px 4px !important;
}

/* ── ─────────────── Custom components ────────────── */

.nav {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    margin: 0 -2.5rem 28px;
    padding: 16px 2.5rem;
    position: sticky; top: 0; z-index: 999;
    display: flex; align-items: center; justify-content: space-between;
}
.nav-left { display: flex; align-items: center; gap: 10px; }
.nav-dot { width: 7px; height: 7px; border-radius: 2px; background: #2563eb; display: inline-block; }
.nav-title { font-size: 0.95rem; font-weight: 700; color: #0f172a; letter-spacing: -0.01em; }
.nav-sep { color: #cbd5e1; font-weight: 400; }
.nav-badge {
    font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.10em;
    color: #94a3b8;
}

.form-sec {
    font-size: 0.66rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: #94a3b8; margin: 20px 0 8px;
}

.dx-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 22px 24px; margin-bottom: 14px;
    display: flex; align-items: flex-start; justify-content: space-between;
}
.dx-eye {
    font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.13em; color: #94a3b8; margin-bottom: 7px;
}
.dx-name {
    font-size: 1.9rem; font-weight: 800; color: #0f172a;
    letter-spacing: -0.02em; line-height: 1.12; margin-bottom: 0;
}
.dx-conf-block { text-align: right; }
.dx-conf-lbl {
    font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.13em; color: #94a3b8; margin-bottom: 7px;
}
.dx-conf-val { font-size: 1.5rem; font-weight: 800; color: #2563eb; font-variant-numeric: tabular-nums; }

.tr-card { border-radius: 9px; padding: 14px 16px; margin: 0 0 18px; display: flex; align-items: flex-start; gap: 10px; }
.tr-dot { width: 9px; height: 9px; border-radius: 50%; margin-top: 5px; flex-shrink: 0; }
.tr-title { font-size: 0.88rem; font-weight: 700; color: #0f172a; margin-bottom: 2px; }
.tr-sub { font-size: 0.76rem; color: #64748b; }
.tr-safe    { background: #f0fdf4; border: 1px solid #bbf7d0; }
.tr-safe .tr-dot    { background: #10b981; }
.tr-unsure  { background: #fffbeb; border: 1px solid #fde68a; }
.tr-unsure .tr-dot  { background: #f59e0b; }
.tr-biopsy  { background: #fef2f2; border: 1px solid #fecaca; }
.tr-biopsy .tr-dot  { background: #ef4444; }

.ch-lbl {
    font-size: 0.66rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.10em; color: #94a3b8; margin: 22px 0 10px;
}

.panel-section-head { display: flex; align-items: baseline; gap: 9px; margin-bottom: 16px; }
.panel-section-icon { font-size: 0.95rem; }
.panel-section-title { font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.11em; color: #0f172a; }
.panel-section-q { font-size: 0.76rem; color: #94a3b8; font-weight: 500; }

.ev-lbl {
    font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: #94a3b8; margin: 18px 0 7px;
}
.ev {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 9px; padding: 12px 16px; margin: 8px 0;
    font-size: 0.82rem; line-height: 1.6; color: #64748b;
}
.ev strong { color: #0f172a; font-weight: 700; }
.ev .str { float: right; font-size: 0.66rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; margin-top: 2px; }
.ev-x {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 9px; padding: 12px 16px; margin: 8px 0;
    font-size: 0.82rem; line-height: 1.6; color: #64748b;
}
.ev-x strong { color: #334155; font-weight: 600; }

.about {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 26px 28px;
}
.empty-icon { font-size: 1.8rem; margin-bottom: 10px; }
.about-h { font-size: 1.05rem; font-weight: 700; color: #0f172a; letter-spacing: -0.01em; margin-bottom: 10px; }
.about-p { font-size: 0.84rem; color: #475569; line-height: 1.8; margin-bottom: 22px; }
.st-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 7px 0; border-bottom: 1px solid #f1f5f9;
    font-size: 0.83rem;
}
.st-row:last-child { border-bottom: none; padding-bottom: 0; }
.st-k { color: #64748b; }
.st-v { color: #0f172a; font-weight: 600; font-variant-numeric: tabular-nums; }
.about-legend { margin-top: 18px; padding-top: 16px; border-top: 1px solid #f1f5f9; }
.about-legend-row { font-size: 0.78rem; color: #475569; line-height: 2.1; }

.footer {
    margin: 48px -2.5rem 0;
    padding: 22px 2.5rem 26px;
    border-top: 1px solid #e2e8f0;
    text-align: center;
}
.footer-brand { font-size: 0.82rem; font-weight: 700; color: #334155; margin-bottom: 6px; }
.footer-brand span { color: #cbd5e1; margin: 0 4px; }
.footer-meta { font-size: 0.76rem; color: #94a3b8; margin-bottom: 4px; }
.footer-disclaimer { font-size: 0.70rem; color: #cbd5e1; }
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
    contra    = float(X_sym["contradiction_severity"].iloc[0]) if "contradiction_severity" in X_sym.columns else 0.0
    fsm_val   = int(X_sym["fsm_state"].iloc[0])       if "fsm_state"    in X_sym.columns else 0
    rec       = BiopsyTriage().recommend(top_certainty=top_cert, conflict_load=conflict, fsm_state=fsm_val,
                                         contradiction_severity=contra)
    FSM_NAMES = ["Evidence Sparse","Hypothesis Forming","Building Evidence","Diagnostic Tension","Resolved"]
    return {
        "disease":     pred_dis,
        "confidence":  confidence,
        "proba":       {DISEASES[i]: float(proba[i]) for i in range(len(DISEASES))},
        "recommendation": rec,
        "top_certainty": top_cert,
        "sym_cert":    {d: float(X_sym[f"certainty_{d}"].iloc[0]) for d in DISEASES if f"certainty_{d}" in X_sym.columns},
        "fsm":         FSM_NAMES[min(fsm_val, 4)],
        "conflict":    conflict,
        "contradiction": contra,
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
                    axis=alt.Axis(format=".0%", labelColor="#94a3b8",
                                  gridColor="#f1f5f9",
                                  domainOpacity=0, tickOpacity=0,
                                  labelFont="Inter", labelFontSize=10)),
            y=alt.Y("Disease:N", sort="-x", title=None,
                    axis=alt.Axis(labelColor="#334155", domainOpacity=0,
                                  tickOpacity=0, labelFont="Inter", labelFontSize=11)),
            tooltip=["Disease:N", alt.Tooltip(f"{x_col}:Q", format=".1%", title=x_col)],
        )
        .properties(height=height, background="transparent")
        .configure_view(strokeWidth=0, fill="transparent")
    )


SEVERITY_TICKS = ["Absent", "Mild", "Moderate", "Severe"]


def severity_slider(label, default=1):
    value = st.slider(label, 0, 3, default)
    pct = value / 3 * 100
    ticks = "".join(
        f'<span class="{"tick-on" if i == value else "tick-off"}">{i} {t}</span>'
        for i, t in enumerate(SEVERITY_TICKS)
    )
    st.markdown(f"""<style>
[data-testid="stSlider"]:has([aria-label="{label}"]) [data-baseweb="slider"] > div > div {{
    background: linear-gradient(to right, #2563eb {pct}%, #e2e8f0 {pct}%) !important;
}}
</style>""", unsafe_allow_html=True)
    st.markdown(f'<div class="tick-row" data-tick-for="{label}">{ticks}</div>', unsafe_allow_html=True)
    return value


def inject_slider_live_sync():
    components.html("""
<script>
const LABELS = ["Erythema", "Scaling", "Definite borders", "Itching"];
function syncSlider(doc, label) {
    const thumb = doc.querySelector('[data-testid="stSlider"] [aria-label="' + label + '"]');
    if (!thumb) return;
    const val = parseFloat(thumb.getAttribute('aria-valuenow'));
    const pct = val / 3 * 100;
    const track = thumb.closest('[data-baseweb="slider"]').querySelector(':scope > div > div');
    if (track) track.style.setProperty('background', 'linear-gradient(to right, #2563eb ' + pct + '%, #e2e8f0 ' + pct + '%)', 'important');
    const tickRow = doc.querySelector('.tick-row[data-tick-for="' + label + '"]');
    if (tickRow) {
        [...tickRow.children].forEach((span, i) => {
            span.className = (i === Math.round(val)) ? 'tick-on' : 'tick-off';
        });
    }
}
function loop() {
    const doc = window.parent.document;
    LABELS.forEach(label => syncSlider(doc, label));
}
setInterval(loop, 30);
</script>
""", height=0)


# ── Navigation bar ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav">
  <div class="nav-left">
    <span class="nav-dot"></span>
    <span class="nav-title">HSCIS-ESD</span>
    <span class="nav-sep">|</span>
    <span class="nav-badge">Biopsy-Free Differential Diagnosis</span>
  </div>
</div>
""", unsafe_allow_html=True)

model, grader, engineer, sym_pipeline, rule_engine = load_model()

left, right = st.columns([1, 1.55], gap="large")

# ── Input panel ────────────────────────────────────────────────────────────────
with left:
    st.markdown('<p class="form-sec">Severity Signs</p>', unsafe_allow_html=True)
    with st.container(border=True):
        erythema         = severity_slider("Erythema")
        scaling          = severity_slider("Scaling")
        definite_borders = severity_slider("Definite borders")
        itching          = severity_slider("Itching")
        inject_slider_live_sync()

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

        # ── Section 1: Model Prediction ─────────────────────
        with st.container(border=True):
            st.markdown("""
<div class="panel-section-head">
  <span class="panel-section-icon">🧬</span>
  <span class="panel-section-title">Model Prediction</span>
  <span class="panel-section-q">— what disease does the pattern match?</span>
</div>""", unsafe_allow_html=True)

            st.markdown(f"""
<div class="dx-card">
  <div>
    <p class="dx-eye">Primary Diagnosis</p>
    <p class="dx-name">{dlabel}</p>
  </div>
  <div class="dx-conf-block">
    <p class="dx-conf-lbl">Classifier Confidence</p>
    <p class="dx-conf-val">{conf*100:.1f}%</p>
  </div>
</div>""", unsafe_allow_html=True)

            prob_df = pd.DataFrame({
                "Disease": [DISEASE_LABELS[d] for d in DISEASES],
                "Probability": [res["proba"][d] for d in DISEASES],
            })
            st.altair_chart(altair_bar(prob_df, "Probability", "#2563eb"), use_container_width=True)

        # ── Section 2: Safety Check ─────────────────────────
        with st.container(border=True):
            st.markdown("""
<div class="panel-section-head">
  <span class="panel-section-icon">🛡️</span>
  <span class="panel-section-title">Safety Check</span>
  <span class="panel-section-q">— is it safe to skip biopsy?</span>
</div>""", unsafe_allow_html=True)

            st.markdown(f"""
<div class="tr-card {tr_cls}">
  <span class="tr-dot"></span>
  <div>
    <p class="tr-title">{tr_title}</p>
    <p class="tr-sub">{tr_sub}</p>
  </div>
</div>""", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Rule Certainty",   f"{res['top_certainty']*100:.1f}%")
            m2.metric("Diagnostic State", res["fsm"].split()[0])
            m3.metric("Conflict Load",    f"{res['conflict']:.3f}")

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
                    tier  = TIER_LABEL.get(tk, tk)
                    signs = ", ".join(tg[tk]["features"])
                    s     = ev_strength(tg[tk]["mc"])
                    st.markdown(f"""<div class="ev"><strong>{tier} {pred_label}</strong><span class="str">{s}</span><br>{signs}</div>""", unsafe_allow_html=True)
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
                    st.markdown(f"""<div class="ev-x"><strong>{sign}</strong> — not typical for {", ".join(diseases)}</div>""", unsafe_allow_html=True)

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
                        tier  = TIER_LABEL.get(tk, tk)
                        signs = ", ".join(g["features"])
                        s     = ev_strength(g["mc"])
                        st.markdown(f"**{tier} {dis}** — *{signs}* (match: {s})")
        else:
            st.caption("No expert rules fired — prediction driven entirely by statistical classifier.")

    else:
        # ── Empty state (pre-diagnosis) ──────────────────────
        st.markdown("""
<div class="about">
  <div class="empty-icon">🩺</div>
  <p class="about-h">Awaiting Clinical Input</p>
  <p class="about-p">
    Fill in the severity and clinical signs on the left, then run diagnosis
    to see the predicted disease, confidence, diagnostic state, and full
    expert-rule reasoning trace.
  </p>
  <div class="about-legend" style="margin-top:4px; padding-top:16px;">
    <div class="about-legend-row">🟢 &nbsp;<strong style="color:#94a3b8;">Safe</strong> &nbsp;— high certainty, low conflict</div>
    <div class="about-legend-row">🟡 &nbsp;<strong style="color:#94a3b8;">Uncertain</strong> &nbsp;— moderate evidence, clinical judgement required</div>
    <div class="about-legend-row">🔴 &nbsp;<strong style="color:#94a3b8;">Biopsy advised</strong> &nbsp;— ambiguous evidence, histopathology recommended</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div class="footer-brand">HSCIS-ESD <span>·</span> Biopsy-Free Differential Diagnosis</div>
  <div class="footer-meta">UCI Dermatology Dataset &nbsp;·&nbsp; 366 patients &nbsp;·&nbsp; 6 ESD classes &nbsp;·&nbsp; CC BY 4.0</div>
  <div class="footer-disclaimer">For research and educational use only — not a substitute for professional clinical diagnosis.</div>
</div>
""", unsafe_allow_html=True)
