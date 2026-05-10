import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

import io
from datetime import datetime
from PIL import Image

# ── ReportLab PDF ─────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, Image as RLImage)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# =============================================================================
# FEATURE NAMES — must match notebook FEATURES list exactly
# =============================================================================
FEATURE_NAMES = [
    "Age", "Gender", "Height", "Weight", "BMI",
    "Systolic BP", "Diastolic BP", "Pulse Pressure",
    "Cholesterol", "Glucose", "Smoke", "Alcohol", "Active",
    "Age Group", "BP Category",
]

# =============================================================================
# LOAD MODEL & SCALER
# =============================================================================
model  = joblib.load("heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")

# =============================================================================
# PAGE CONFIG & GLOBAL CSS
# =============================================================================
st.set_page_config(
    page_title="CardioAI — Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, .main, [data-testid="stAppViewContainer"] {
    background: #060810 !important;
    color: #e8eaf2;
    font-family: 'DM Sans', sans-serif;
}
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1200px; margin: 0 auto; }

.hero {
    position: relative; text-align: center;
    padding: 3.5rem 2rem 2.5rem; margin-bottom: 2rem; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(220,50,50,.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: .7rem; font-weight: 600; letter-spacing: .2em;
    text-transform: uppercase; color: #e05555; margin-bottom: .6rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.4rem, 5vw, 3.6rem);
    line-height: 1.1; color: #f5f6fa; margin: 0 0 .8rem;
}
.hero-title span { color: #e05555; font-style: italic; }
.hero-sub {
    font-size: 1rem; color: #7a7f9a; max-width: 560px; margin: 0 auto;
    font-weight: 300; line-height: 1.6;
}
.hero-line {
    width: 60px; height: 2px;
    background: linear-gradient(90deg,#e05555,#ff8c6b);
    margin: 1.6rem auto 0; border-radius: 2px;
}
.card {
    background: #0d1020; border: 1px solid #1e2138;
    border-radius: 16px; padding: 1.6rem 1.8rem;
    margin-bottom: 1rem; position: relative; overflow: hidden;
}
.card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(224,85,85,.4), transparent);
}
.card-label {
    font-size: .68rem; font-weight: 600; letter-spacing: .15em;
    text-transform: uppercase; color: #e05555;
    margin-bottom: 1.2rem; display: flex; align-items: center; gap: 8px;
}
.card-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg,#e05555 0%,transparent 100%); opacity: .25;
}
[data-testid="stSlider"] > div > div > div > div { background: #e05555 !important; }
[data-testid="stSlider"] label, [data-testid="stRadio"] label,
[data-testid="stSelectbox"] label {
    font-size: .82rem !important; color: #9198b8 !important; font-weight: 400 !important;
}
div[data-baseweb="select"] > div {
    background: #111525 !important; border: 1px solid #1e2138 !important; border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #c0392b 0%, #e74c3c 50%, #c0392b 100%) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 16px 0 !important;
    font-size: 1rem !important; font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important; width: 100% !important;
    box-shadow: 0 4px 24px rgba(224,85,85,.35) !important; transition: all .25s !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 32px rgba(224,85,85,.55) !important; transform: translateY(-1px) !important;
}
.result-high {
    background: linear-gradient(135deg,rgba(192,57,43,.12),rgba(231,76,60,.06));
    border: 1px solid rgba(231,76,60,.45); border-radius: 16px;
    padding: 2rem; text-align: center; margin-bottom: 1rem;
}
.result-low {
    background: linear-gradient(135deg,rgba(39,174,96,.12),rgba(46,204,113,.06));
    border: 1px solid rgba(46,204,113,.45); border-radius: 16px;
    padding: 2rem; text-align: center; margin-bottom: 1rem;
}
.result-icon { font-size: 2.4rem; margin-bottom: .4rem; }
.result-title { font-family: 'DM Serif Display', serif; font-size: 1.8rem; margin-bottom: .3rem; }
.result-pct { font-size: .9rem; color: #9198b8; }
.rec-pill {
    display: flex; align-items: flex-start; gap: 10px;
    background: #111525; border: 1px solid #1e2138; border-radius: 10px;
    padding: 10px 14px; margin-bottom: 7px; font-size: .88rem;
    color: #c8cce0; line-height: 1.4;
}
.rec-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #e05555; margin-top: 5px; flex-shrink: 0;
}
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg,transparent,#1e2138 20%,#1e2138 80%,transparent);
    margin: 2rem 0;
}
[data-testid="stDownloadButton"] > button {
    background: transparent !important; border: 1px solid #e05555 !important;
    color: #e05555 !important; border-radius: 10px !important;
    padding: 10px 24px !important; font-weight: 500 !important; transition: all .2s !important;
}
[data-testid="stDownloadButton"] > button:hover { background: rgba(224,85,85,.1) !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HERO
# =============================================================================
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-powered cardiovascular diagnostics</div>
    <h1 class="hero-title">Cardio<span>AI</span></h1>
    <p class="hero-sub">
        Early heart disease detection powered by machine learning.
        Enter patient data and receive a personalized risk report in seconds.
    </p>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER: derived features
# =============================================================================
def compute_derived(age, height, weight, ap_hi, ap_lo):
    bmi           = round(weight / (height / 100) ** 2, 2)
    pulse_pressure= ap_hi - ap_lo

    if age < 40:   age_group_enc = 0
    elif age < 50: age_group_enc = 1
    elif age < 60: age_group_enc = 2
    else:          age_group_enc = 3

    if ap_hi < 120 and ap_lo < 80:    bp_category = 0
    elif ap_hi < 130 and ap_lo < 80:  bp_category = 1
    elif ap_hi < 140 or ap_lo < 90:   bp_category = 2
    else:                              bp_category = 3

    return bmi, pulse_pressure, age_group_enc, bp_category

# =============================================================================
# PDF GENERATOR
# =============================================================================
def generate_pdf(inputs_dict, prediction, probability, recs,
                 feature_names, feature_impact) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=1.8*cm, bottomMargin=1.8*cm,
                            leftMargin=2*cm, rightMargin=2*cm)

    RED   = colors.HexColor("#c0392b")
    DARK  = colors.HexColor("#0d0f1a")
    MID   = colors.HexColor("#1a1d30")
    MID2  = colors.HexColor("#14172a")
    LIGHT = colors.HexColor("#304492")
    GREY  = colors.HexColor("#6c7199")
    GREEN = colors.HexColor("#1e8449")
    RBG_H = colors.HexColor("#2c0a0a")
    RBGL  = colors.HexColor("#0a2c12")

    base = getSampleStyleSheet()["Normal"]
    def sty(name, **kw):
        return ParagraphStyle(name, parent=base, **kw)

    sec_sty  = sty("H",  fontSize=9, fontName="Helvetica-Bold", textColor=RED, spaceBefore=12, spaceAfter=5)
    body_sty = sty("B",  fontSize=8.5, textColor=LIGHT, leading=13, spaceAfter=3)
    small_sty= sty("SM", fontSize=7, textColor=GREY, leading=11, spaceAfter=2)

    story = []

    age = inputs_dict['age']
    sex = "Male" if inputs_dict['gender'] == 1 else "Female"

    header_data = [[
        Paragraph("<b><font size=20 color='#e74c3c'>CardioAI</font></b><br/>"
                  "<font size=8 color='#6c7199'>Cardiovascular Risk Report</font>",
                  sty("HH", alignment=TA_LEFT, leading=18)),
        Paragraph(f"<font size=7.5 color='#6c7199'>Generated: "
                  f"{datetime.now().strftime('%d %b %Y  %H:%M')}<br/>"
                  f"Patient Age: {age} y/o  |  Sex: {sex}</font>",
                  sty("HHR", alignment=TA_RIGHT, leading=13)),
    ]]
    ht = Table(header_data, colWidths=["55%","45%"])
    ht.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),DARK),
        ("TOPPADDING",(0,0),(-1,-1),12),("BOTTOMPADDING",(0,0),(-1,-1),12),
        ("LEFTPADDING",(0,0),(0,-1),14),("RIGHTPADDING",(-1,0),(-1,-1),14),
        ("LINEBELOW",(0,-1),(-1,-1),1.5,RED),
    ]))
    story += [ht, Spacer(1,12)]

    is_high   = prediction == 1
    res_color = RED if is_high else GREEN
    res_bg    = RBG_H if is_high else RBGL
    res_hex   = "e74c3c" if is_high else "27ae60"
    res_text  = "HIGH RISK DETECTED" if is_high else "LOW RISK"
    res_data  = [[
        Paragraph(f"<b><font size=14 color='#{res_hex}'>{'⚠  ' if is_high else '✔  '}{res_text}</font></b>",
                  sty("RB", alignment=TA_CENTER)),
        Paragraph(f"<font size=10 color='#{res_hex}'>Risk probability: <b>{probability:.1%}</b></font>",
                  sty("RP", alignment=TA_CENTER)),
    ]]
    rt = Table(res_data, colWidths=["50%","50%"])
    rt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),res_bg),
        ("BOX",(0,0),(-1,-1),1.5,res_color),
        ("TOPPADDING",(0,0),(-1,-1),11),("BOTTOMPADDING",(0,0),(-1,-1),11),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story += [rt, Spacer(1,12)]

    story.append(Paragraph("PATIENT DATA", sec_sty))
    chol_l = {1:"Normal", 2:"Above Normal", 3:"Well Above Normal"}
    gluc_l = {1:"Normal", 2:"Above Normal", 3:"Well Above Normal"}
    bp_l   = {0:"Normal", 1:"Elevated", 2:"High Stage 1", 3:"High Stage 2"}
    ag_l   = {0:"< 40", 1:"40–49", 2:"50–59", 3:"60+"}

    def td(t, bold=False):
        fn = "Helvetica-Bold" if bold else "Helvetica"
        hex_c = "e8eaf2" if bold else "6c7199"
        return Paragraph(f"<font name='{fn}' size=8 color='#{hex_c}'>{t}</font>",
                         sty(f"td{hash(t)%9999}"))

    rows = [
        ["Age",            f"{inputs_dict['age']} years",   "Sex",           sex],
        ["Height",         f"{inputs_dict['height']} cm",   "Weight",        f"{inputs_dict['weight']} kg"],
        ["BMI",            f"{inputs_dict['bmi']:.1f}",     "Age Group",     ag_l[inputs_dict['age_group_enc']]],
        ["Systolic BP",    f"{inputs_dict['ap_hi']} mmHg",  "Diastolic BP",  f"{inputs_dict['ap_lo']} mmHg"],
        ["Pulse Pressure", f"{inputs_dict['pulse_pressure']} mmHg","BP Category",f"{bp_l[inputs_dict['bp_category']]}"],
        ["Cholesterol",    chol_l[inputs_dict['cholesterol']], "Glucose",    gluc_l[inputs_dict['gluc']]],
        ["Smoker",         "Yes" if inputs_dict['smoke'] else "No","Alcohol","Yes" if inputs_dict['alco'] else "No"],
        ["Active",         "Yes" if inputs_dict['active'] else "No", "", ""],
    ]
    tbl_data = [[td(l1), td(v1, bold=True), td(l2), td(v2, bold=True)] for l1,v1,l2,v2 in rows]
    pt = Table(tbl_data, colWidths=[3.8*cm,4.5*cm,3.8*cm,4.5*cm])
    pt.setStyle(TableStyle([
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[MID,MID2]),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
        ("LINEBELOW",(0,-1),(-1,-1),0.5,RED),
    ]))
    story += [pt, Spacer(1,12)]

    story.append(Paragraph("FEATURE IMPACT", sec_sty))
    sorted_idx = np.argsort(feature_impact)[::-1][:8]
    fi_rows = [[
        Paragraph("<b><font size=7.5 color='#e8eaf2'>Feature</font></b>", sty("fh1")),
        Paragraph("<b><font size=7.5 color='#e8eaf2'>Impact</font></b>", sty("fh2", alignment=TA_CENTER)),
        Paragraph("<b><font size=7.5 color='#e8eaf2'>Visual</font></b>", sty("fh3")),
    ]]
    for i in sorted_idx:
        bar = "█" * max(1, int(feature_impact[i]) // 2)
        fi_rows.append([
            td(feature_names[i]),
            Paragraph(f"<b><font size=8 color='#e74c3c'>{feature_impact[i]:.1f}%</font></b>",
                      sty(f"fv{i}", alignment=TA_CENTER)),
            Paragraph(f"<font size=5.5 color='#e74c3c'>{bar}</font>", sty(f"fb{i}")),
        ])
    fit = Table(fi_rows, colWidths=[4.5*cm, 2*cm, 10.1*cm])
    fit.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),RED),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[MID,MID2]),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),7),
        ("LINEBELOW",(0,-1),(-1,-1),0.5,RED),
    ]))
    story += [fit, Spacer(1,12)]

    story.append(Paragraph("RECOMMENDATIONS", sec_sty))
    for r in recs:
        story.append(Paragraph(f"•  {r}", body_sty))
    story.append(Spacer(1,8))

    story.append(HRFlowable(width="100%", thickness=0.5, color=RED))
    story.append(Spacer(1,4))
    story.append(Paragraph(
        "CardioAI — ECU Data Mining 2026  |  "
        "DISCLAIMER: This report is AI-assisted screening only and does "
        "not constitute a medical diagnosis. Always consult a qualified cardiologist.",
        small_sty))

    doc.build(story)
    return buf.getvalue()


# =============================================================================
# INPUTS
# =============================================================================
st.markdown("<div class='card'><div class='card-label'>Patient information</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1: age    = st.slider("Age (years)", 20, 80, 50)
with c2: height = st.slider("Height (cm)", 140, 210, 170)
with c3: weight = st.slider("Weight (kg)", 40, 180, 70)
with c4:
    sex     = st.radio("Gender", ["Female", "Male"], horizontal=True)
    gender  = 1 if sex == "Male" else 2
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'><div class='card-label'>Blood pressure</div>", unsafe_allow_html=True)
c5, c6 = st.columns(2)
with c5: ap_hi = st.slider("Systolic BP (mmHg)", 80, 200, 120)
with c6: ap_lo = st.slider("Diastolic BP (mmHg)", 40, 140, 80)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'><div class='card-label'>Lab results & lifestyle</div>", unsafe_allow_html=True)
c7, c8, c9 = st.columns(3)
with c7:
    cholesterol = st.selectbox("Cholesterol level", [1, 2, 3],
        format_func=lambda x: {1:"Normal", 2:"Above Normal", 3:"Well Above Normal"}[x])
with c8:
    gluc = st.selectbox("Glucose level", [1, 2, 3],
        format_func=lambda x: {1:"Normal", 2:"Above Normal", 3:"Well Above Normal"}[x])
with c9:
    smoke  = st.radio("Smoker",   ["No", "Yes"], horizontal=True)
    smoke  = 1 if smoke == "Yes" else 0

c10, c11 = st.columns(2)
with c10:
    alco   = st.radio("Alcohol intake", ["No", "Yes"], horizontal=True)
    alco   = 1 if alco == "Yes" else 0
with c11:
    active = st.radio("Physically active", ["No", "Yes"], horizontal=True)
    active = 1 if active == "Yes" else 0
st.markdown("</div>", unsafe_allow_html=True)

# Compute derived features
bmi, pulse_pressure, age_group_enc, bp_category = compute_derived(age, height, weight, ap_hi, ap_lo)

# Show computed values
st.markdown("<div class='card'><div class='card-label'>Auto-computed features</div>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
bp_labels = {0:"Normal 🟢", 1:"Elevated 🟡", 2:"High Stage 1 🟠", 3:"High Stage 2 🔴"}
ag_labels  = {0:"< 40", 1:"40–49", 2:"50–59", 3:"60+"}
m1.metric("BMI",            f"{bmi:.1f}")
m2.metric("Pulse Pressure", f"{pulse_pressure} mmHg")
m3.metric("Age Group",      ag_labels[age_group_enc])
m4.metric("BP Category",    bp_labels[bp_category])
st.markdown("</div>", unsafe_allow_html=True)

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_btn = st.button("🫀  Analyze & Predict Risk")

# =============================================================================
# RESULTS
# =============================================================================
if predict_btn:
    # Build feature vector — order must match FEATURES in notebook
    data_in = [
        age, gender, height, weight, bmi,
        ap_hi, ap_lo, pulse_pressure,
        cholesterol, gluc, smoke, alco, active,
        age_group_enc, bp_category,
    ]
    data_arr    = np.array([data_in])
    data_scaled = scaler.transform(data_arr)
    prediction  = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    feat_imp = model.feature_importances_ * np.abs(data_scaled[0])
    feat_imp = feat_imp / feat_imp.sum() * 100

    is_high = prediction == 1
    recs = (
        ["See a cardiologist immediately",
         "Follow prescribed medications strictly",
         "Reduce cholesterol and salt intake",
         "Avoid smoking and alcohol",
         "Manage stress and get adequate sleep",
         "Monitor heart rate and blood pressure daily"]
        if is_high else
        ["Schedule an annual cardiovascular checkup",
         "Maintain a healthy and balanced diet",
         "Exercise aerobically for at least 30 minutes daily",
         "Stay hydrated and reduce caffeine intake",
         "Maintain a healthy body weight",
         "Monitor blood pressure regularly"]
    )

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    col_g, col_r = st.columns(2)
    with col_g:
        gc = "#e74c3c" if is_high else "#27ae60"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(probability * 100, 1),
            title={"text": "RISK SCORE", "font": {"color": "#9198b8", "size": 12}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#3a3f5c", "tickfont": {"color": "#9198b8", "size": 10}},
                "bar": {"color": gc, "thickness": .25},
                "bgcolor": "#0d1020", "bordercolor": "#1e2138", "borderwidth": 1,
                "steps": [
                    {"range": [0, 40],   "color": "#0a1f12"},
                    {"range": [40, 65],  "color": "#1f1a0a"},
                    {"range": [65, 100], "color": "#1f0a0a"},
                ],
                "threshold": {"value": 50, "line": {"color": "#e0e0e0", "width": 1.5}, "thickness": .8},
            },
            number={"suffix": "%", "font": {"color": gc, "size": 36}},
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "DM Sans"}, height=280,
            margin=dict(t=30, b=10, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        if is_high:
            st.markdown(f"""
            <div class="result-high">
                <div class="result-icon">⚠️</div>
                <div class="result-title" style="color:#e74c3c">High risk detected</div>
                <div class="result-pct">Probability: <b style="color:#e74c3c">{probability:.1%}</b></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
                <div class="result-icon">✅</div>
                <div class="result-title" style="color:#27ae60">Low risk</div>
                <div class="result-pct">Probability: <b style="color:#27ae60">{probability:.1%}</b></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='font-size:.72rem;color:#7a7f9a;margin-bottom:.6rem;"
                    "letter-spacing:.1em;text-transform:uppercase'>Recommendations</div>",
                    unsafe_allow_html=True)
        for r in recs:
            st.markdown(f"<div class='rec-pill'><div class='rec-dot'></div>{r}</div>",
                        unsafe_allow_html=True)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='card'><div class='card-label'>Feature impact analysis</div>",
                unsafe_allow_html=True)
    sidx     = np.argsort(feat_imp)
    mean_val = np.mean(feat_imp)
    fig2 = go.Figure(go.Bar(
        x=feat_imp[sidx],
        y=[FEATURE_NAMES[i] for i in sidx],
        orientation="h",
        marker=dict(
            color=["#e74c3c" if feat_imp[i] > mean_val else "#3498db" for i in sidx],
            line=dict(width=0)
        ),
        text=[f"{v:.1f}%" for v in feat_imp[sidx]],
        textposition="outside",
        textfont=dict(color="#9198b8", size=11),
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9198b8", family="DM Sans"), height=420,
        margin=dict(t=10, b=10, l=10, r=60),
        xaxis=dict(title="Impact (%)", gridcolor="#1a1d30", zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1a1d30", tickfont=dict(size=11)),
        bargap=0.35,
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
    with st.spinner("Generating PDF report…"):
        inputs_dict = {
            "age": age, "gender": gender, "height": height, "weight": weight,
            "bmi": bmi, "ap_hi": ap_hi, "ap_lo": ap_lo,
            "pulse_pressure": pulse_pressure, "cholesterol": cholesterol,
            "gluc": gluc, "smoke": smoke, "alco": alco, "active": active,
            "age_group_enc": age_group_enc, "bp_category": bp_category,
        }
        pdf_bytes = generate_pdf(
            inputs_dict=inputs_dict,
            prediction=prediction, probability=probability, recs=recs,
            feature_names=FEATURE_NAMES, feature_impact=feat_imp,
        )

    _, dl_col, _ = st.columns([1, 1, 1])
    with dl_col:
        st.download_button(
            label="📥  Download full PDF report",
            data=pdf_bytes,
            file_name=f"cardioai_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )