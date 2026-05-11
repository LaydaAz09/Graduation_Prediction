"""
Sistem Prediksi Kelulusan Mahasiswa
Machine Learning-Powered Graduation Prediction with Random Forest

Author: Layda
BINUS University - Data Science
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import warnings
import plotly 
warnings.filterwarnings('ignore')

# PAGE CONFIGURATION

    
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS STYLING

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* ── FIX GAP UTAMA ── */
           .block-container {
        padding-top: 3rem !important;
    }
    
    [data-testid="stAppViewContainer"] > .main {
        padding-top: 2rem !important;
    }
    
    .grad-header {
        margin-top: 1.5rem !important;
    }
       [data-testid="stAppViewContainer"] > .main {
        padding-top: 3.5rem !important;
    }
    /* ── TAMPILKAN HAMBURGER DI MOBILE ── */
    @media (max-width: 768px) {
        header[data-testid="stHeader"] {
            height: auto !important;
            min-height: 3rem !important;
            max-height: 3rem !important;
            visibility: visible !important;
            display: flex !important;
            background: #0f0b20 !important;
            border-bottom: 1px solid rgba(139,92,246,0.2) !important;
        }
        [data-testid="stHeader"] button,
        [data-testid="stHeader"] svg {
            color: #a78bfa !important;
            fill: #a78bfa !important;
        }
    }
    @media (min-width: 769px) {
    section[data-testid="stSidebar"] {
        display: flex !important;
        visibility: visible !important;
    }
}

    /* ── FIX SIDEBAR GAP ── */
    [data-testid="stSidebarContent"] {
    padding: 0.8rem 1.1rem 1.4rem 1.1rem !important;
}
    [data-testid="stSidebarContent"] {
        padding: 0.8rem 1.1rem 1.4rem 1.1rem !important;
        margin-top: 0 !important;
    }

    /* ── SIDEBAR STYLE ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1235 0%, #0f0b20 100%) !important;
        border-right: 1px solid rgba(139,92,246,0.12);
    }
    [data-testid="stSidebar"] * { color: #c9bfef !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(139,92,246,0.18) !important; margin: 0.75rem 0 !important; }

    /* ── SIDEBAR SELECT ── */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(139,92,246,0.13) !important;
        border: 1.5px solid rgba(139,92,246,0.35) !important;
        border-radius: 10px !important;
        color: #e2d9f3 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #a78bfa !important; }

    /* ── SIDEBAR RADIO - FIXED ── */
    [data-testid="stSidebar"] .stRadio > div {
        display: flex !important;
        flex-direction: column !important;
        gap: 3px !important;
    }
    [data-testid="stSidebar"] .stRadio > label,
    [data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"] {
        display: none !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
        padding: 0.52rem 0.9rem !important;
        border-radius: 9px !important;
        transition: all 0.15s !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        color: #b8aee0 !important;
        background: transparent !important;
        border: none !important;
        width: 100% !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(139,92,246,0.12) !important;
        color: #e2d9f3 !important;
    }
    [data-testid="stSidebar"] .stRadio label > div:first-child {
        display: none !important;
    }
    [data-testid="stSidebar"] .stRadio label > div:last-child,
    [data-testid="stSidebar"] .stRadio label > p,
    [data-testid="stSidebar"] .stRadio label span {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        color: #b8aee0 !important;
        font-size: 0.84rem !important;
    }

    /* ── HEADER ── */
    .grad-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 60%, #a855f7 100%);
        padding: 2.2rem 2.2rem; border-radius: 1.1rem; margin-bottom: 1.5rem;
        color: white; box-shadow: 0 8px 32px rgba(118,75,162,0.35);
        position: relative; overflow: hidden;
    }
    .grad-header::before {
        content: ''; position: absolute; top: -50%; right: -8%;
        width: 380px; height: 380px; border-radius: 50%;
        background: rgba(255,255,255,0.07); pointer-events: none;
    }
    .grad-header h1 {
        font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800;
        color: white !important; margin: 0 0 0.3rem 0; position: relative;
    }
    .grad-header p { color: rgba(255,255,255,0.82); font-size: 0.92rem; margin: 0; position: relative; }

    /* ── STAT CARDS ── */
     .stat-card {
        background: #1a1235; border-radius: 0.85rem; padding: 1.2rem 1rem;
        text-align: center; box-shadow: 0 2px 16px rgba(0,0,0,0.4);
        border: 1px solid rgba(139,92,246,0.2); transition: transform 0.2s;
    }
    .stat-card:hover { transform: translateY(-3px); }
    .stat-card .sc-icon  { font-size: 1.5rem; margin-bottom: 0.25rem; }
    .stat-card .sc-val   { font-family: 'Syne', sans-serif; font-size: 1.7rem; font-weight: 800; color: #a78bfa; }
    .stat-card .sc-label { font-size: 0.68rem; color: #9d8ec7; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.15rem; }

    /* ── IPK CARDS ── */
    .ipk-good { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.75rem; border-radius: 1rem; color: white; text-align: center; box-shadow: 0 8px 28px rgba(118,75,162,0.35); }
    .ipk-mid  { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.75rem; border-radius: 1rem; color: white; text-align: center; box-shadow: 0 8px 28px rgba(245,87,108,0.3); }
    .ipk-low  { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.75rem; border-radius: 1rem; color: white; text-align: center; box-shadow: 0 8px 28px rgba(79,172,254,0.3); }
    .ipk-num  { font-family: 'Syne', sans-serif; font-size: 4.2rem; font-weight: 800; line-height: 1; }

    /* ── RESULT CARDS ── */
    .res-lulus { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2.5rem; border-radius: 1.1rem; color: white; text-align: center; box-shadow: 0 10px 36px rgba(118,75,162,0.35); position: relative; overflow: hidden; }
    .res-tidak { background: linear-gradient(#f093fb 0%, #f5576c 100%); padding: 2.5rem; border-radius: 1.1rem; color: white; text-align: center; box-shadow: 0 10px 36px rgba(79,172,254,0.35); position: relative; overflow: hidden; }
    .res-lulus::before, .res-tidak::before { content: ''; position: absolute; top: -40%; right: -8%; width: 280px; height: 280px; border-radius: 50%; background: rgba(255,255,255,0.07); pointer-events: none; }
    .res-title { font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800; color:white; position:relative; }
    .res-stats { display:flex; justify-content:center; gap:3rem; margin-top:1.5rem; position:relative; }
    .res-stat-val   { font-family:'Syne',sans-serif; font-size:2.1rem; font-weight:800; color:white; }
    .res-stat-label { font-size:0.7rem; color:rgba(255,255,255,0.75); text-transform:uppercase; letter-spacing:0.06em; }

    /* ── SCORE BADGE ── */
    .score-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1.5px solid rgba(255,255,255,0.35);
        border-radius: 999px;
        padding: 0.15rem 0.9rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: rgba(255,255,255,0.9);
        letter-spacing: 0.04em;
        margin-top: 0.3rem;
    }

    /* ── METRIC BOXES ── */
    .mbox { background: #1a1235; border-radius: 0.85rem; padding: 1.2rem; text-align: center; border: 1px solid rgba(139,92,246,0.2); box-shadow: 0 2px 12px rgba(0,0,0,0.4); transition: transform 0.2s; }
    .mbox:hover { transform: translateY(-3px); }
    .mbox-icon { font-size: 1.4rem; margin-bottom: 0.35rem; }
    .mbox-val  { font-family:'Syne',sans-serif; font-size:1.55rem; font-weight:800; color:#a78bfa; }
    .mbox-lbl  { font-size:0.7rem; color:#9d8ec7; text-transform:uppercase; letter-spacing:0.05em; margin-top:0.15rem; }

    /* ── PREDICTION SCORE BOX ── */
    .score-box {
        background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(168,85,247,0.15) 100%);
        border: 1.5px solid rgba(139,92,246,0.4);
        border-radius: 0.85rem; padding: 1.2rem; text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.4); transition: transform 0.2s;
    }
    .score-box:hover { transform: translateY(-3px); }
    .score-box-icon { font-size: 1.4rem; margin-bottom: 0.35rem; }
    .score-box-val  { font-family:'Syne',sans-serif; font-size:1.55rem; font-weight:800; color:#a78bfa; }
    .score-box-lbl  { font-size:0.7rem; color:#9d8ec7; text-transform:uppercase; letter-spacing:0.05em; margin-top:0.15rem; }

    /* ── BUTTON ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important; border: none !important;
        border-radius: 0.6rem !important; padding: 0.7rem 2rem !important;
        font-weight: 700 !important; font-size: 0.95rem !important;
        box-shadow: 0 4px 16px rgba(118,75,162,0.35) !important; transition: all 0.2s !important;
    }
    .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(118,75,162,0.45) !important; }

    /* ── SIDEBAR LOGO & MISC ── */
    .logo-icon-box {
        width: 42px; height: 42px;
        background: linear-gradient(135deg, #667eea 0%, #a855f7 100%);
        border-radius: 11px; display: flex; align-items: center; justify-content: center;
        font-size: 1.25rem; margin-bottom: 0.55rem;
        box-shadow: 0 4px 14px rgba(139,92,246,0.5);
    }
    .sb-logo-title {
        font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 800;
        color: white; line-height: 1.25; margin: 0;
    }
    .sb-logo-sub {
        font-size: 0.68rem; color: rgba(201,191,239,0.55); margin-top: 1px;
    }
    .status-badge {
        display: flex; align-items: center; gap: 0.5rem;
        background: rgba(6,214,160,0.1); border: 1px solid rgba(6,214,160,0.25);
        border-radius: 8px; padding: 0.38rem 0.85rem; margin: 0.5rem 0;
        width: 100%;
    }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #06d6a0; box-shadow: 0 0 7px #06d6a0; animation: pulse 2s infinite; flex-shrink: 0; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
    .status-text { font-size: 0.78rem; color: #06d6a0 !important; font-weight: 600; }
    .sb-section {
        font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.12em;
        color: rgba(201,191,239,0.38) !important; font-weight: 700;
        margin: 0.8rem 0 0.35rem 0; display: block; padding-left: 0.1rem;
    }
    .sb-mode-hint {
        font-size: 0.67rem; color: rgba(201,191,239,0.4) !important;
        margin-bottom: 0.3rem; display: block; padding-left: 0.1rem;
    }

    /* ── ABOUT CARDS ── */
    .about-card { background: #1a1235; border-radius: 0.85rem; padding: 1.4rem; border: 1px solid rgba(139,92,246,0.25); box-shadow: 0 2px 12px rgba(0,0,0,0.4); }
    .about-card h3 { font-family:'Syne',sans-serif; font-size:0.97rem; font-weight:700; color:#a78bfa; margin-bottom:0.7rem; }
    .about-card p, .about-card li { font-size:0.84rem; color:#c9bfef; line-height:1.65; }
    .formula-box { background: #0f0b20; border-radius: 0.6rem; padding: 0.8rem 1rem; font-size: 0.8rem; color: #a78bfa; line-height: 1.8; border: 1px solid rgba(139,92,246,0.3); margin-top: 0.5rem; font-family: monospace; }
    div[data-baseweb="select"] { border-radius: 0.6rem !important; }

    /* ── EXPANDER TEXT PUTIH ── */
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span,
    [data-testid="stExpander"] div,
    [data-testid="stExpander"] li,
    .streamlit-expanderContent p,
    .streamlit-expanderContent span,
    .streamlit-expanderContent div {
        color: #ffffff !important;
    }
            
    /* ── DARK BACKGROUND ── */
    .stApp { background-color: #0f0b20 !important; }
    [data-testid="stAppViewContainer"] { background-color: #0f0b20 !important; }
    [data-testid="stMain"] { background-color: #0f0b20 !important; }
    .main { background-color: #0f0b20 !important; }
    section.main > div { background-color: #0f0b20 !important; }
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] > div,
    [data-testid="stFileUploader"] > div > div,
    [data-testid="stFileUploader"] > div > div > div,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] > div,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section > div,
    [data-testid="stFileUploader"] li,
    [data-testid="stFileUploader"] ul {
        background-color: #1a1235 !important;
        border-color: rgba(139,92,246,0.3) !important;
        border-radius: 0.75rem !important;
        color: #e2d9f3 !important;
    }
    [data-testid="stFileUploader"] * {
        color: #e2d9f3 !important;
        background-color: transparent !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #0f0b20 !important;
        border: 1px solid rgba(139,92,246,0.35) !important;
        color: #a78bfa !important;
        border-radius: 8px !important;
    }

    /* ── HEADING & TEKS UTAMA ── */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffffff !important;
    }
      /* ── LABEL FORM PUTIH ── */
        label p, label span, label div,
        .stSelectbox label, .stNumberInput label,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span,
        [data-testid="stWidgetLabel"] div,
        .stSelectbox > label,
        .stNumberInput > label {
            color: #ffffff !important;
        }
        /* ── EXPANDER HEADER GELAP ── */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] details > summary,
    div[data-testid="stExpander"] > details > summary {
        background-color: #1a1235 !important;
        border: 1px solid rgba(139,92,246,0.3) !important;
        border-radius: 10px !important;
        color: #e2d9f3 !important;
    }
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary svg {
        color: #e2d9f3 !important;
        fill: #a78bfa !important;
    }

    /* ── EXPANDER CONTAINER GELAP ── */
    [data-testid="stExpander"],
    [data-testid="stExpander"] details {
        background-color: #1a1235 !important;
        border: 1px solid rgba(139,92,246,0.2) !important;
        border-radius: 10px !important;
    }
    /* ── DATAFRAME / TABEL GELAP ── */
    [data-testid="stDataFrame"] > div,
    [data-testid="stDataFrame"] iframe,
    .stDataFrame > div {
        background-color: #1a1235 !important;
        border: 1px solid rgba(139,92,246,0.2) !important;
        border-radius: 10px !important;
    }
    [data-testid="stDataFrame"] th {
        background-color: #0f0b20 !important;
        color: #a78bfa !important;
        border-color: rgba(139,92,246,0.2) !important;
    }
    [data-testid="stDataFrame"] td {
        background-color: #1a1235 !important;
        color: #ffffff !important;
        border-color: rgba(139,92,246,0.15) !important;
    }
    [data-testid="stDataFrame"] tr:hover td {
        background-color: rgba(139,92,246,0.12) !important;
    }
    /* ── FILE UPLOADER BUTTON ── */
    [data-testid="stFileUploaderDropzoneInput"] + div button,
    [data-testid="baseButton-secondary"] {
        background-color: #1a1235 !important;
        color: #a78bfa !important;
        border: 1px solid rgba(139,92,246,0.35) !important;
        border-radius: 8px !important;
    }
    /* ── FILE UPLOADER BUTTON ── */
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] button:hover,
    [data-testid="stFileUploader"] button:focus,
    button[kind="secondary"],
    .stFileUploader button {
        background-color: #1a1235 !important;
        color: #a78bfa !important;
        border: 1px solid rgba(139,92,246,0.35) !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploader"] button span,
    [data-testid="stFileUploader"] button p {
        color: #a78bfa !important;
    }
    [data-testid="stFileUploader"] svg {
        fill: #a78bfa !important;
        stroke: #a78bfa !important;
    }
    /* ── FILE UPLOADER AFTER UPLOAD ── */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploader"] [class*="uploadedFile"],
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] > div > div {
        background-color: #1a1235 !important;
        border: 1px solid rgba(139,92,246,0.3) !important;
        border-radius: 10px !important;
        color: #e2d9f3 !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] p,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] small {
        color: #e2d9f3 !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button {
        background-color: transparent !important;
        border: none !important;
        color: #a78bfa !important;
    }
    /* ── HAPUS WRAPPER LUAR FILE UPLOADER ── */
    [data-testid="stFileUploader"] > div:first-child,
    [data-testid="stFileUploader"] > div:first-child > div {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
        box-shadow: none !important;
        margin: 0 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #1a1235 !important;
        border: 1px solid rgba(139,92,246,0.3) !important;
        border-radius: 0.75rem !important;
        padding: 0.6rem 1rem !important;
        min-height: unset !important;
    }
    /* ── FILE UPLOADER ── */
    [data-testid="stFileUploader"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #1a1235 !important;
        border: 1px solid rgba(139,92,246,0.3) !important;
        border-radius: 0.75rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 1rem !important;
        min-height: 60px !important;
    }
    [data-testid="stFileUploaderDropzone"] > div {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 1rem !important;
        width: 100% !important;
    }
    [data-testid="stFileUploader"] * { color: #e2d9f3 !important; }
    [data-testid="stFileUploader"] button {
        background-color: #0f0b20 !important;
        border: 1px solid rgba(139,92,246,0.35) !important;
        color: #a78bfa !important;
        border-radius: 8px !important;
    }
    /* ── BUTTON TEXT PUTIH ── */
    .stButton > button,
    .stButton > button p,
    .stButton > button span {
        color: #ffffff !important;
    }
        /* ── HEADER ATAS SAMAKAN DENGAN BACKGROUND ── */
    header[data-testid="stHeader"] {
        background-color: #0f0b20 !important;
    }

    .stAppHeader {
        background-color: #0f0b20 !important;
    }

    [data-testid="stToolbar"] {
        background-color: #0f0b20 !important;
    }

    [data-testid="stDecoration"] {
        background: #0f0b20 !important;
    }
        /* ── ICON & TEXT HEADER JADI PUTIH ── */
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] button svg,
    header[data-testid="stHeader"] a,
    header[data-testid="stHeader"] p,
    header[data-testid="stHeader"] span,
    header[data-testid="stHeader"] div {
        color: white !important;
        fill: white !important;
    }
</style>
""", unsafe_allow_html=True)


# SESSION STATE

defaults = {
    'active_menu': 'Input Data',
    'prediction_done': False,
    'result': {},
    'csv_results': None,
    'input_mode': 'Input Data Personal',
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v



# LOAD DATA & TRAIN MODEL

@st.cache_data
def load_data():
    try:
        dtype_map = {'ips_1': str,'ips_2': str,'ips_3': str,'ips_4': str,'ips_5': str,'ips_6': str,'IPK': str}
        df = pd.read_excel('dataset_mahasiswa_clean.xlsx', delimiter=';', dtype=dtype_map, low_memory=False, encoding='utf-8-sig')
        for col in ['ips_1','ips_2','ips_3','ips_4','ips_5','ips_6','IPK']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False).astype(float)
        df['lulus_tepat_waktu'] = (df['lulus_tepat_waktu'] == 'Tepat waktu').astype(int)
        return df.dropna(), None
    except FileNotFoundError:
        return None, "not_found"
    except Exception as e:
        return None, str(e)


@st.cache_resource
def train_model():
    df, err = load_data()
    if err:
        return None, None, None, None, None, 0.0, 0, err, {}

    le_prodi, le_gender, le_status = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df = df.copy()
    df['prodi_enc']  = le_prodi.fit_transform(df['prodi'])
    df['gender_enc'] = le_gender.fit_transform(df['jenis_kelamin'])
    df['status_enc'] = le_status.fit_transform(df['status_pegawai'])

    features = ['prodi_enc','gender_enc','status_enc','umur','ips_1','ips_2','ips_3','ips_4','ips_5','ips_6']
    X, y = df[features], df['lulus_tepat_waktu']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    param_grid = {
        'n_estimators'     : [100],
        'max_depth'        :  [20],
        'min_samples_split': [10],
        'min_samples_leaf' : [4],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    base_rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1)

    grid_search = GridSearchCV(
        estimator  = base_rf,
        param_grid = param_grid,
        cv         = cv,
        scoring    = 'f1_weighted',
        n_jobs     = 1,
        verbose    = 0,
        refit      = True,
    )
    grid_search.fit(X_train_s, y_train)

    model       = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score  = grid_search.best_score_

    accuracy = accuracy_score(y_test, model.predict(X_test_s))
    return model, scaler, le_prodi, le_gender, le_status, accuracy, len(df), None, best_params



# PREDICTION HELPER

def do_predict(model, scaler, le_prodi, le_gender, le_status, model_ok,
               prodi, jenis_kelamin, status_pegawai, umur,
               ips_1, ips_2, ips_3, ips_4, ips_5, ips_6):
    ips_vals  = [ips_1, ips_2, ips_3, ips_4, ips_5, ips_6]
    ipk       = float(np.mean(ips_vals))
    ips_trend = ips_6 - ips_1
    ips_std   = float(np.std(ips_vals))

    # ── ATURAN UTAMA: IPK < 2.0 → pasti Tidak Lulus (confidence 100%) ──
    if ipk < 2.0:
        return dict(pred=0, confidence=100.0,
                    prob_tepat=0.0, prob_tidak=100.0,
                    ipk=ipk, ips_vals=ips_vals,
                    ips_trend=ips_trend, ips_std=ips_std, prodi=prodi,
                    gender=jenis_kelamin, status=status_pegawai, umur=umur)

    if not model_ok:
        # ── DEMO MODE ──
        if ipk >= 3.0:
            pred = 1
            conf = float(min(70.0 + (ipk - 3.0) / 1.0 * 25.0, 97.0))
        elif ipk >= 2.5:
            pred = 1
            conf = float(60.0 + (ipk - 2.5) / 0.5 * 10.0)
        else:
            pred = 1
            conf = float(55.0 + (ipk - 2.0) / 0.5 * 5.0)
        prob_tepat = conf if pred == 1 else 100.0 - conf
        prob_tidak = 100.0 - prob_tepat

    else:
        pe = le_prodi.transform([prodi])[0]
        ge = le_gender.transform([jenis_kelamin])[0]
        se = le_status.transform([status_pegawai])[0]
        inp_s = scaler.transform(np.array([[pe, ge, se, umur, ips_1, ips_2, ips_3, ips_4, ips_5, ips_6]]))
        pred  = int(model.predict(inp_s)[0])
        proba = model.predict_proba(inp_s)[0]  # [prob_tidak, prob_tepat]
        prob_tidak = float(proba[0] * 100)
        prob_tepat = float(proba[1] * 100)
        conf  = prob_tepat if pred == 1 else prob_tidak

        # ── OVERRIDE ──
        if pred == 0 and ipk >= 2.0:
            proba_lulus = float(model.predict_proba(inp_s)[0][1])
            if proba_lulus > 0.35:
                pred = 1
                prob_tepat = proba_lulus * 100
                prob_tidak = 100.0 - prob_tepat
                conf = prob_tepat

    return dict(pred=pred, confidence=conf,
                prob_tepat=prob_tepat, prob_tidak=prob_tidak,
                ipk=ipk, ips_vals=ips_vals,
                ips_trend=ips_trend, ips_std=ips_std, prodi=prodi,
                gender=jenis_kelamin, status=status_pegawai, umur=umur)



# SHARED HELPER: CSV result stats

def csv_stats(df_res):
    mask_lulus = df_res['Kategori'].str.startswith('✅')
    lulus      = int(mask_lulus.sum())
    tidak      = int(len(df_res) - lulus)
    total      = len(df_res)
    grad_rate  = lulus / total * 100 if total > 0 else 0
    avg_ipk    = float(df_res['IPK'].mean())
    ipk_lulus  = float(df_res[mask_lulus]['IPK'].mean())  if lulus > 0 else 0.0
    ipk_tidak  = float(df_res[~mask_lulus]['IPK'].mean()) if tidak > 0 else 0.0
    return lulus, tidak, total, grad_rate, avg_ipk, ipk_lulus, ipk_tidak



# PAGE SECTIONS

def dark_table(df):
    html = "<table style='width:100%;border-collapse:collapse;border-radius:10px;overflow:hidden;'>"
    html += "<tr>" + "".join(
        f"<th style='background:#0f0b20;color:#a78bfa;padding:10px 14px;text-align:left;border-bottom:1px solid rgba(139,92,246,0.3);'>{col}</th>"
        for col in df.columns
    ) + "</tr>"
    for _, row in df.iterrows():
        html += "<tr>" + "".join(
            f"<td style='background:#1a1235;color:#ffffff;padding:9px 14px;border-bottom:1px solid rgba(139,92,246,0.12);'>{val}</td>"
            for val in row
        ) + "</tr>"
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)
    
def render_header(accuracy, total_data, model_ok, best_params):
    st.markdown("""
    <div class="grad-header" style="display:flex;align-items:center;gap:2rem;padding:1.8rem 2.2rem;">
        <div style="font-size:5rem;flex-shrink:0;line-height:1;">🎓</div>
        <div style="position:relative;">
            <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.15em;
                        color:rgba(255,255,255,0.65);margin-bottom:0.4rem;font-weight:600;">
                BINUS University · Data Science
            </div>
            <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                        color:white;line-height:1.2;margin-bottom:0.3rem;">
                Graduation Prediction System
            </div>
            <div style="font-size:0.85rem;color:rgba(255,255,255,0.75);font-weight:400;">
                Machine Learning · Random Forest · Prediksi Kelulusan Mahasiswa
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def page_input(model, scaler, le_prodi, le_gender, le_status, model_ok):
    st.markdown("## 📝 Input Data Mahasiswa")
    col_form, col_preview = st.columns([1.1, 0.9])

    with col_form:
        st.markdown("### 📋 Data Pribadi")
        prodi_opts  = le_prodi.classes_.tolist()  if model_ok else ['Teknik Informatika','Sistem Informasi','Manajemen','Akuntansi','Ilmu Komunikasi','Psikologi','Hukum']
        gender_opts = le_gender.classes_.tolist() if model_ok else ['Laki-laki','Perempuan']
        status_opts = le_status.classes_.tolist() if model_ok else ['Bekerja','Tidak Bekerja']

        prodi          = st.selectbox("🎓 Program Studi", prodi_opts)
        ca, cb         = st.columns(2)
        jenis_kelamin  = ca.selectbox("👤 Jenis Kelamin", gender_opts)
        status_pegawai = cb.selectbox("💼 Status Pekerjaan", status_opts)
        umur           = st.number_input("🎂 Umur (tahun)", 17, 50, 22)

        st.markdown("### 📈 IPS per Semester")
        st.info("💡 Masukkan IPS Semester 1–6. IPK akan dihitung otomatis.")
        c1s, c2s, c3s = st.columns(3)
        with c1s:
            ips_1 = st.number_input("📘 Semester 1", 0.0, 4.0, 3.22, 0.01, key="s1")
            ips_4 = st.number_input("📘 Semester 4", 0.0, 4.0, 2.91, 0.01, key="s4")
        with c2s:
            ips_2 = st.number_input("📗 Semester 2", 0.0, 4.0, 3.43, 0.01, key="s2")
            ips_5 = st.number_input("📗 Semester 5", 0.0, 4.0, 3.33, 0.01, key="s5")
        with c3s:
            ips_3 = st.number_input("📙 Semester 3", 0.0, 4.0, 2.95, 0.01, key="s3")
            ips_6 = st.number_input("📙 Semester 6", 0.0, 4.0, 3.38, 0.01, key="s6")

    ips_vals = [ips_1, ips_2, ips_3, ips_4, ips_5, ips_6]
    ipk      = float(np.mean(ips_vals))

    with col_preview:
        if ipk >= 3.0:   ipk_cls, ipk_note = "ipk-good", "✅ Prestasi Sangat Baik"
        elif ipk >= 2.5: ipk_cls, ipk_note = "ipk-good", "✅ Prestasi Baik"
        elif ipk >= 2.0: ipk_cls, ipk_note = "ipk-mid",  "📊 Cukup Baik"
        else:            ipk_cls, ipk_note = "ipk-low",  "📊 Perlu Peningkatan"

        st.markdown(f'<div class="{ipk_cls}">'
                    f'<div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;opacity:.82;margin-bottom:.4rem;">IPK KUMULATIF</div>'
                    f'<div class="ipk-num">{ipk:.2f}</div>'
                    f'<div style="opacity:.78;font-size:.82rem;margin-top:.3rem;">dari 4.00</div>'
                    f'<div style="margin-top:.7rem;font-weight:700;font-size:.95rem;">{ipk_note}</div>'
                    f'</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 📈 Tren IPS per Semester")
        bar_colors = ['#667eea' if v >= 3.0 else '#f093fb' if v >= 2.0 else '#4facfe' for v in ips_vals]
        fig_bar = go.Figure(go.Bar(
            x=[f'Sem {i+1}' for i in range(6)], y=ips_vals,
            marker_color=bar_colors, text=[f"{v:.2f}" for v in ips_vals], textposition='outside'))
        fig_bar.add_hline(y=ipk, line_dash="dash", line_color="#764ba2", line_width=2,
                          annotation_text=f"IPK: {ipk:.2f}", annotation_position="right")
        fig_bar.update_layout(height=260, yaxis=dict(range=[0,4.6]),
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              margin=dict(l=5,r=50,t=15,b=5), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    if st.button("🔮 PREDIKSI KELULUSAN SEKARANG", use_container_width=True):
        result = do_predict(model, scaler, le_prodi, le_gender, le_status, model_ok,
                            prodi, jenis_kelamin, status_pegawai, umur,
                            ips_1, ips_2, ips_3, ips_4, ips_5, ips_6)
        st.session_state.prediction_done = True
        st.session_state.result = result
        st.success("✅ Prediksi berhasil! Buka **📊 Hasil Prediksi** di sidebar.")


def page_hasil_personal():
    if not st.session_state.prediction_done:
        st.warning("⚠️ Belum ada prediksi. Isi data di **📝 Input Data** terlebih dahulu.")
        return
    r = st.session_state.result
    pred, conf, ipk, ips_trend = r['pred'], r['confidence'], r['ipk'], r['ips_trend']
    prob_tepat = r.get('prob_tepat', conf if pred == 1 else 100.0 - conf)
    prob_tidak = r.get('prob_tidak', 100.0 - prob_tepat)

    st.markdown("## 📊 Hasil Prediksi Kelulusan")
    cls   = "res-lulus" if pred == 1 else "res-tidak"
    emoji = "✅" if pred == 1 else "❌"
    title = "LULUS TEPAT WAKTU" if pred == 1 else "TIDAK LULUS TEPAT WAKTU"

    #  Hasil prediksi + prediction score dalam satu card 
    st.markdown(
        f'<div class="{cls}">'
        f'<div style="font-size:3rem;position:relative;">{emoji}</div>'
        f'<div class="res-title">{title}</div>'
        f'<div class="res-stats">'
        f'<div style="text-align:center;"><div class="res-stat-val">{ipk:.2f}</div><div class="res-stat-label">IPK</div></div>'
        f'<div style="text-align:center;"><div class="res-stat-val">{ips_trend:+.2f}</div><div class="res-stat-label">Tren IPS</div></div>'
        f'<div style="text-align:center;border-left:1px solid rgba(255,255,255,0.25);padding-left:3rem;">'
        f'<div class="res-stat-val">{prob_tepat:.1f}%</div>'
        f'<div class="res-stat-label">Peluang Tepat Waktu</div>'
        f'</div>'
        f'<div style="text-align:center;">'
        f'<div class="res-stat-val">{prob_tidak:.1f}%</div>'
        f'<div class="res-stat-label">Peluang Tidak Tepat</div>'
        f'</div>'
        f'</div></div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Metric boxes: tambah Prediction Score 
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="mbox"><div class="mbox-icon">📊</div>'
                    f'<div class="mbox-val">{ipk:.2f}</div><div class="mbox-lbl">IPK</div></div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="mbox"><div class="mbox-icon">📈</div>'
                    f'<div class="mbox-val">{max(r["ips_vals"]):.2f}</div><div class="mbox-lbl">IPS Tertinggi</div></div>',
                    unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="mbox"><div class="mbox-icon">📉</div>'
                    f'<div class="mbox-val">{min(r["ips_vals"]):.2f}</div><div class="mbox-lbl">IPS Terendah</div></div>',
                    unsafe_allow_html=True)
    with col4:
        score_icon  = "🎯" if pred == 1 else "⚠️"
        score_color = "#06d6a0" if pred == 1 else "#f093fb"
        score_val   = prob_tepat if pred == 1 else prob_tidak
        score_lbl   = "Score Tepat Waktu" if pred == 1 else "Score Tidak Tepat"
        st.markdown(
            f'<div class="score-box">'
            f'<div class="score-box-icon">{score_icon}</div>'
            f'<div class="score-box-val" style="color:{score_color};">{score_val:.1f}%</div>'
            f'<div class="score-box-lbl">{score_lbl}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("### 📋 Ringkasan Input")
        rows = [("Program Studi", r['prodi']), ("Jenis Kelamin", r['gender']),
                ("Status Pekerjaan", r['status']), ("Umur", f"{r['umur']} tahun"),
                ("IPK", f"{ipk:.2f}")] + \
               [(f"IPS Semester {i+1}", f"{v:.2f}") for i, v in enumerate(r['ips_vals'])]
        dark_table(pd.DataFrame(rows, columns=["Variabel","Nilai"]))
    with cb:
        st.markdown("### 🎯 Prediction Score Detail")
        # Gauge bar visualisasi probability
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Bar(
            x=[prob_tepat, prob_tidak],
            y=["Tepat Waktu", "Tidak Tepat Waktu"],
            orientation='h',
            marker_color=['#667eea', '#f093fb'],
            text=[f"{prob_tepat:.1f}%", f"{prob_tidak:.1f}%"],
            textposition='inside',
            textfont=dict(size=15, color='white', family='Syne'),
        ))
        fig_prob.add_vline(x=50, line_dash="dash", line_color="rgba(255,255,255,0.4)", line_width=1.5)
        fig_prob.update_layout(
            height=180,
            xaxis=dict(range=[0, 100], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor='rgba(26,18,53,1)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            font=dict(color='#ffffff'),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # Tabel ringkasan score
        score_rows = [
            ("Peluang Tepat Waktu",    f"{prob_tepat:.2f}%"),
            ("Peluang Tidak Tepat",    f"{prob_tidak:.2f}%"),
            ("Hasil Prediksi",         "✅ Tepat Waktu" if pred == 1 else "❌ Tidak Tepat Waktu"),
            ("Confidence Model",       f"{conf:.2f}%"),
        ]
        dark_table(pd.DataFrame(score_rows, columns=["Metrik", "Nilai"]))


def page_analisis_personal(model, model_ok):
    if not st.session_state.prediction_done:
        st.warning("⚠️ Belum ada prediksi. Isi data di **📝 Input Data** terlebih dahulu.")
        return
    r = st.session_state.result
    ipk, ips_vals, ips_trend, ips_std = r['ipk'], r['ips_vals'], r['ips_trend'], r['ips_std']

    st.markdown("## 🔍 Analisis Detail Faktor Akademik")
    cg, cb = st.columns(2)
    with cg:
        st.markdown("### 📊 Gauge IPK")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=ipk,
            delta={'reference': 3.0, 'valueformat': '.2f'}, number={'valueformat': '.2f'},
            gauge={'axis': {'range': [0,4]}, 'bar': {'color': "#7c3aed"},
                   'steps': [{'range':[0,2.0],'color':'#fce4ec'},{'range':[2.0,2.75],'color':'#ede9ff'},
                              {'range':[2.75,3.5],'color':'#e8f5e9'},{'range':[3.5,4.0],'color':'#e3f2fd'}],
                   'threshold': {'line': {'color':'#f5576c','width':3}, 'thickness':0.8, 'value':2.0}}))
        fig_g.update_layout(height=280, margin=dict(l=20,r=20,t=40,b=10), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_g, use_container_width=True)
        if ipk < 2.0:    st.error(f"**{ipk:.2f}** — 🚨 Perlu peningkatan segera")
        elif ipk < 2.75: st.warning(f"**{ipk:.2f}** — ⚠️ Di bawah rata-rata ideal")
        elif ipk < 3.0:  st.info(f"**{ipk:.2f}** — 📊 Cukup, target ≥ 3.0")
        elif ipk < 3.5:  st.success(f"**{ipk:.2f}** — ✅ Baik")
        else:             st.success(f"**{ipk:.2f}** — 🌟 Sangat Baik!")

    with cb:
        st.markdown("### 📈 IPS per Semester")
        bar_colors = ['#667eea' if v>=3.0 else '#f093fb' if v>=2.0 else '#4facfe' for v in ips_vals]
        fig_b = go.Figure(go.Bar(x=[f'Sem {i+1}' for i in range(6)], y=ips_vals,
                                 marker_color=bar_colors, text=[f"{v:.2f}" for v in ips_vals], textposition='outside'))
        fig_b.add_hline(y=ipk, line_dash="dash", line_color="#764ba2",
                        annotation_text=f"IPK {ipk:.2f}", annotation_position="right")
        fig_b.update_layout(height=280, yaxis=dict(range=[0,4.6]),
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=5,r=50,t=15,b=5), showlegend=False)
        st.plotly_chart(fig_b, use_container_width=True)
        if ips_trend > 0.5:    st.success(f"📈 Tren Positif (+{ips_trend:.2f})")
        elif ips_trend > 0:    st.info(f"📊 Tren Meningkat (+{ips_trend:.2f})")
        elif ips_trend > -0.3: st.warning(f"📉 Tren Menurun ({ips_trend:.2f})")
        else:                  st.error(f"🚨 Tren Negatif ({ips_trend:.2f})")
        if ips_std < 0.3:   st.success(f"🎯 Sangat Stabil (SD: {ips_std:.2f})")
        elif ips_std < 0.5: st.info(f"📊 Cukup Stabil (SD: {ips_std:.2f})")
        else:               st.warning(f"⚠️ Fluktuatif (SD: {ips_std:.2f})")


def page_rekomendasi_personal():
    if not st.session_state.prediction_done:
        st.warning("⚠️ Belum ada prediksi. Isi data di **📝 Input Data** terlebih dahulu.")
        return
    r = st.session_state.result
    ipk, ips_trend, ips_std = r['ipk'], r['ips_trend'], r['ips_std']
    is_working = 'Bekerja' in r['status']

    st.markdown("## 💡 Rekomendasi Tindakan")
    recs = []
    if ipk < 2.75:
        recs.append(("🚨 PRIORITAS UTAMA — Tingkatkan Prestasi Akademik",
                     ["Konsultasi segera dengan dosen pembimbing akademik.",
                      "Identifikasi mata kuliah dengan nilai terendah.",
                      "Ikuti program remedial atau bimbingan belajar intensif.",
                      "Kurangi aktivitas non-akademik untuk sementara waktu.",
                      "Manfaatkan office hours dosen untuk bertanya langsung."]))
    elif ipk < 3.0:
        recs.append(("⚠️ PERLU PERHATIAN — Optimalkan Prestasi",
                     ["Target IPK 3.0 untuk zona sangat aman.",
                      "Bergabung dengan study group atau tutor sebaya.",
                      "Konsultasi rutin dengan dosen pembimbing setiap bulan.",
                      "Aktif bertanya dan berdiskusi di kelas."]))
    else:
        recs.append(("✅ PRESTASI BAIK — Pertahankan!",
                     ["IPK sudah baik! Pertahankan konsistensinya.",
                      "Fokus pada penyelesaian studi tepat waktu.",
                      "Manfaatkan prestasi untuk beasiswa atau program unggulan."]))
    if ips_trend < -0.3:
        recs.append(("📉 TREN MENURUN — Evaluasi Penyebab",
                     ["Identifikasi mata kuliah penyebab nilai turun.",
                      "Evaluasi manajemen waktu dan metode belajar.",
                      "Buat jadwal belajar yang lebih terstruktur."]))
    elif ips_trend > 0.5:
        recs.append(("🌟 TREN POSITIF — Pertahankan Momentum!",
                     ["Tren sangat baik — pertahankan metode belajar ini!",
                      "Jadikan momentum ini motivasi semester depan."]))
    if is_working and ipk < 3.0:
        recs.append(("💼 BEKERJA + IPK PERLU PENINGKATAN",
                     ["Pertimbangkan mengurangi jam kerja di semester kritis.",
                      "Prioritaskan jadwal kuliah dan ujian di atas kerja."]))
    if ips_std > 0.5:
        recs.append(("⚠️ NILAI FLUKTUATIF — Tingkatkan Konsistensi",
                     ["Buat rutinitas belajar tetap setiap minggu.",
                      "Hindari sistem belajar kebut semalam.",
                      "Review materi berkala minimal 2x seminggu."]))
    recs.append(("💪 KESEHATAN & MENTAL",
                 ["Istirahat cukup (7–8 jam/hari) dan olahraga rutin.",
                  "Kelola stres dengan relaksasi atau hobi positif.",
                  "Jangan ragu cari bantuan konseling akademik jika perlu."]))

    for title, items in recs:
        with st.expander(f"**{title}**", expanded=True):
            for item in items:
                st.markdown(f"✓ {item}")


def page_visualisasi_personal():
    if not st.session_state.prediction_done:
        st.warning("⚠️ Belum ada prediksi. Isi data di **📝 Input Data** terlebih dahulu.")
        return
    r = st.session_state.result
    ipk, ips_vals = r['ipk'], r['ips_vals']
    ips_trend, ips_std, conf = r['ips_trend'], r['ips_std'], r['confidence']

    st.markdown("## 📈 Visualisasi & Perbandingan")
    fig_cmp = go.Figure()
    cats  = ['IPK','IPS Tertinggi','IPS Terendah']
    yours = [ipk, max(ips_vals), min(ips_vals)]
    fig_cmp.add_trace(go.Bar(name='Nilai Anda', x=cats, y=yours, marker_color='#667eea',
                              text=[f"{v:.2f}" for v in yours], textposition='outside'))
    fig_cmp.add_trace(go.Bar(name='Standar Minimum', x=cats, y=[2.0,2.0,2.0], marker_color='#f5576c',
                              text=["2.00","2.00","2.00"], textposition='outside'))
    fig_cmp.add_trace(go.Bar(name='Standar Ideal', x=cats, y=[3.0,3.5,2.75], marker_color='#06d6a0',
                              text=["3.00","3.50","2.75"], textposition='outside'))
    fig_cmp.update_layout(barmode='group', height=400, yaxis=dict(range=[0,4.8]),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='#ffffff'),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02,xanchor="right", x=1,font=dict(color='#ffffff'),bgcolor='rgba(0,0,0,0)'), margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig_cmp, use_container_width=True)

    cr, ci = st.columns([2,1])
    with cr:
        st.markdown("### 🎯 Radar Profil Mahasiswa")
        cats_r = ['IPK','Tren IPS','Konsistensi','IPS Terakhir']
        vals_r = [(ipk/4)*100, min(100,max(0,((ips_trend+1)/2)*100)),
                  max(0,(1-ips_std/0.5))*100, conf, (ips_vals[5]/4)*100]
        fig_r = go.Figure(go.Scatterpolar(
            r=vals_r+[vals_r[0]], theta=cats_r+[cats_r[0]],
            fill='toself', fillcolor='rgba(102,126,234,0.35)',
            line=dict(color='#764ba2',width=2.5), marker=dict(size=9,color='#a855f7')))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),
                            showlegend=False, height=360,
                            paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=40,r=40,t=40,b=40))
        st.plotly_chart(fig_r, use_container_width=True)
    with ci:
        st.info("**Area luas = semakin baik**\n\n"
                "- **IPK** — Prestasi keseluruhan\n"
                "- **Tren IPS** — Perkembangan nilai\n"
                "- **Konsistensi** — Kestabilan nilai\n"
                "- **IPS Terakhir** — Performa Sem 6")

    st.markdown("---")
    summary = [
        ("IPK", f"{ipk:.2f}", "✅ Baik" if ipk>=3.0 else ("📊 Cukup" if ipk>=2.0 else "⚠️ Perlu Peningkatan")),
        ("IPS Tertinggi", f"{max(ips_vals):.2f}", "✅ Baik" if max(ips_vals)>=3.5 else "📊 Cukup"),
        ("IPS Terendah",  f"{min(ips_vals):.2f}", "✅ Aman" if min(ips_vals)>=2.0 else "⚠️ Rendah"),
        ("Tren Sem6–Sem1",f"{ips_trend:+.2f}", "📈 Naik" if ips_trend>0 else "📉 Turun"),
        ("Konsistensi (SD)", f"{ips_std:.2f}", "🎯 Stabil" if ips_std<0.3 else ("📊 Cukup" if ips_std<0.5 else "⚠️ Fluktuatif")),
    ]
    dark_table(pd.DataFrame(summary, columns=["Aspek","Nilai","Status"]))


def page_upload_csv(model, scaler, le_prodi, le_gender, le_status, model_ok):
    st.markdown("## 📤 Upload & Prediksi CSV")
    st.info("**Format CSV**\n"
            "(`nama`, `prodi`, `jenis_kelamin`, `status_pegawai`, `ips_1`–`ips_6`, `IPK`, `umur`)")

    st.markdown("<p style='color:#ffffff;font-size:1rem;font-weight:600;margin-top:1rem;margin-bottom:0.3rem;'>Upload file CSV mahasiswa</p>", unsafe_allow_html=True)
    uploaded = st.file_uploader("x", type=["csv"], label_visibility="hidden")
    if uploaded is not None:
        try:
            try:
                df_up = pd.read_csv(uploaded, delimiter=';')
                if df_up.shape[1] < 5:
                    uploaded.seek(0)
                    df_up = pd.read_csv(uploaded, delimiter=',')
            except Exception:
                uploaded.seek(0)
                df_up = pd.read_csv(uploaded, delimiter=',')

            for col in ['ips_1','ips_2','ips_3','ips_4','ips_5','ips_6','IPK']:
                 if col in df_up.columns:
                     df_up[col] = (df_up[col].astype(str)
                      .str.replace('"','', regex=False)
                      .str.replace(',','.', regex=False)
                      .str.strip()
                      .astype(float))
        
            df_up.columns = df_up.columns.str.strip().str.lower()
            for col in ['ips_1','ips_2','ips_3','ips_4','ips_5','ips_6']:
                if col in df_up.columns:
                    df_up[col] = df_up[col].astype(str).str.replace(',','.', regex=False).astype(float)

            missing = [c for c in ['ips_1','ips_2','ips_3','ips_4','ips_5','ips_6'] if c not in df_up.columns]
            if missing:
                st.error(f"❌ Kolom tidak ditemukan: {', '.join(missing)}")
            else:
                st.success(f"✅ Dataset berhasil dimuat: **{len(df_up)} mahasiswa**")
                with st.expander("👁️ Preview Data"):
                    dark_table(df_up.head())

                if st.button("🔮 ANALISIS SEMUA MAHASISWA", use_container_width=True):
                    results_list = []
                    prog = st.progress(0)
                    for i, row in df_up.iterrows():
                        pv = str(row.get('prodi', le_prodi.classes_[0] if model_ok else 'Teknik Informatika'))
                        gv = str(row.get('jenis_kelamin', le_gender.classes_[0] if model_ok else 'Laki-laki'))
                        sv = str(row.get('status_pegawai', le_status.classes_[0] if model_ok else 'Tidak Bekerja'))
                        uv = float(row.get('umur', 22))
                        nv = str(row.get('nama', f'Mahasiswa {i+1}'))
                        if model_ok:
                            if pv not in le_prodi.classes_:  pv = le_prodi.classes_[0]
                            if gv not in le_gender.classes_: gv = le_gender.classes_[0]
                            if sv not in le_status.classes_: sv = le_status.classes_[0]

                        r = do_predict(model, scaler, le_prodi, le_gender, le_status, model_ok,
                                       pv, gv, sv, uv,
                                       float(row['ips_1']), float(row['ips_2']),
                                       float(row['ips_3']), float(row['ips_4']),
                                       float(row['ips_5']), float(row['ips_6']))
                        results_list.append({
                            'Nama':                    nv,
                            'IPK':                     round(r['ipk'], 2),
                            'Kategori':                "✅ Lulus Tepat Waktu" if r['pred']==1 else "❌ Tidak Lulus Tepat Waktu",
                            '% Tepat Waktu':           f"{r.get('prob_tepat', r['confidence']):.1f}%",
                            '% Tidak Tepat Waktu':     f"{r.get('prob_tidak', 100-r['confidence']):.1f}%",
                            'Tren IPS':                f"{r['ips_trend']:+.2f}",
                        })
                        prog.progress((i+1)/len(df_up))

                    prog.empty()
                    st.session_state.csv_results = pd.DataFrame(results_list)
                    st.success(f"✅ Selesai! {len(df_up)} mahasiswa dianalisis. Lihat hasilnya di sidebar →")
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")

    if st.session_state.csv_results is None:
        st.markdown("---")
        st.info("📂 Upload file CSV dan klik **Analisis** untuk memulai.")


def page_hasil_csv():
    if st.session_state.csv_results is None:
        st.warning("⚠️ Belum ada data. Upload CSV di menu **📤 Upload & Prediksi** terlebih dahulu.")
        return

    df_res = st.session_state.csv_results
    lulus, tidak, total, grad_rate, avg_ipk, ipk_lulus, ipk_tidak = csv_stats(df_res)

    st.markdown("## 📊 Hasil Prediksi")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 60%,#a855f7 100%);
                padding:2rem 2.2rem;border-radius:1.1rem;margin-bottom:1.5rem;
                color:white;box-shadow:0 8px 32px rgba(118,75,162,0.35);
                position:relative;overflow:hidden;">
        <div style="position:absolute;top:-50%;right:-5%;width:280px;height:280px;
                    border-radius:50%;background:rgba(255,255,255,0.07);pointer-events:none;"></div>
        <div style="font-size:.71rem;font-weight:600;letter-spacing:.08em;opacity:.82;
                    margin-bottom:.5rem;text-transform:uppercase;">
            📊 Hasil Analisis data beberapa Mahasiswa — {total} Mahasiswa
        </div>
        <div style="display:flex;gap:3rem;align-items:flex-end;position:relative;flex-wrap:wrap;">
            <div>
                <div style="font-family:'Syne',sans-serif;font-size:4rem;font-weight:800;line-height:1;">{grad_rate:.0f}%</div>
                <div style="font-size:.82rem;opacity:.82;margin-top:.2rem;">Graduation Rate</div>
            </div>
            <div style="display:flex;gap:2rem;padding-bottom:.4rem;flex-wrap:wrap;">
                <div style="text-align:center;">
                    <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;">{lulus}</div>
                    <div style="font-size:.72rem;opacity:.78;text-transform:uppercase;letter-spacing:.06em;">Lulus Tepat</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;">{tidak}</div>
                    <div style="font-size:.72rem;opacity:.78;text-transform:uppercase;letter-spacing:.06em;">Tidak Tepat</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;">{avg_ipk:.2f}</div>
                    <div style="font-size:.72rem;opacity:.78;text-transform:uppercase;letter-spacing:.06em;">Avg IPK</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for col, icon, val, lbl, clr in zip(
        st.columns(4), ["👥","✅","❌","📊"],
        [total, lulus, tidak, f"{avg_ipk:.2f}"],
        ["Total Mahasiswa","Lulus Tepat Waktu","Tidak Tepat Waktu","Rata-rata IPK"],
        ["#7c3aed","#667eea","#4facfe","#a855f7"]
    ):
        with col:
            st.markdown(f'<div class="stat-card"><div class="sc-icon">{icon}</div>'
                        f'<div class="sc-val" style="color:{clr};">{val}</div>'
                        f'<div class="sc-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Tabel Hasil Prediksi Lengkap")
    df_display = df_res.copy().reset_index(drop=True)
    df_display.insert(0, 'No', range(1, len(df_display)+1))
    dark_table(df_display)

    st.markdown("---")
    st.markdown("### 📊 Ringkasan Statistik")
    cs, cd = st.columns(2)
    with cs:
        stats = [("Total Mahasiswa", str(total)),
                 ("Lulus Tepat Waktu", f"{lulus} ({grad_rate:.1f}%)"),
                 ("Tidak Lulus Tepat Waktu", f"{tidak} ({100-grad_rate:.1f}%)"),
                 ("Rata-rata IPK Keseluruhan", f"{avg_ipk:.2f}"),
                 ("IPK Tertinggi", f"{df_res['IPK'].max():.2f}"),
                 ("IPK Terendah",  f"{df_res['IPK'].min():.2f}"),
                 ("IPK Std Deviasi", f"{df_res['IPK'].std():.2f}")]
        dark_table(pd.DataFrame(stats, columns=["Metrik","Nilai"]))
    with cd:
        cats_stats = [
            ("Avg IPK — Lulus Tepat Waktu",   f"{ipk_lulus:.2f}"),
            ("Avg IPK — Tidak Tepat Waktu",   f"{ipk_tidak:.2f}"),
            ("IPK ≥ 3.0 (Baik)",              f"{(df_res['IPK']>=3.0).sum()} mahasiswa"),
            ("IPK 2.0–2.99 (Cukup)",          f"{((df_res['IPK']>=2.0)&(df_res['IPK']<3.0)).sum()} mahasiswa"),
            ("IPK < 2.0 (Risiko)",            f"{(df_res['IPK']<2.0).sum()} mahasiswa"),
        ]
        dark_table(pd.DataFrame(cats_stats, columns=["Metrik","Nilai"]))

    st.markdown("<br>", unsafe_allow_html=True)
    csv_out = df_display.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Hasil CSV", csv_out,
                       f"hasil_prediksi_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                       "text/csv", use_container_width=True)


def page_visualisasi_csv():
    if st.session_state.csv_results is None:
        st.warning("⚠️ Belum ada data. Upload CSV di menu **📤 Upload & Prediksi** terlebih dahulu.")
        return

    df_res = st.session_state.csv_results
    lulus, tidak, total, grad_rate, avg_ipk, ipk_lulus, ipk_tidak = csv_stats(df_res)

    st.markdown("## 📈 Visualisasi Data Beberapa Mahasiswa")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🍩 Distribusi Prediksi")
        fig_donut = go.Figure(go.Pie(
            labels=['✅ Lulus Tepat Waktu','❌ Tidak Tepat Waktu'],
            values=[lulus, tidak], hole=0.55,
            marker=dict(colors=['#667eea','#4facfe'], line=dict(color='white',width=3)),
            textinfo='percent', textfont=dict(size=14,color='white')))
        fig_donut.add_annotation(text=f"<b>{grad_rate:.0f}%</b><br>Lulus",
                                 x=0.5, y=0.5, font=dict(size=18,color='#7c3aed'), showarrow=False)
        fig_donut.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)',
                                showlegend=True, legend=dict(orientation="h",y=-0.1),
                                margin=dict(l=10,r=10,t=10,b=30))
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        st.markdown("#### 📊 Graduation Rate")
        fig_hbar = go.Figure(go.Bar(
            y=['❌ Tidak Tepat Waktu','✅ Lulus Tepat Waktu'],
            x=[tidak/total*100, grad_rate], orientation='h',
            marker_color=['#4facfe','#667eea'],
            text=[f"{tidak/total*100:.1f}%", f"{grad_rate:.1f}%"],
            textposition='inside', textfont=dict(size=15,color='white',family='Syne')))
        fig_hbar.update_layout(height=320, xaxis=dict(range=[0,105],showgrid=False,zeroline=False),
                               yaxis=dict(showgrid=False), plot_bgcolor='#1a1235',
                               paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#c9bfef'),
                               margin=dict(l=10,r=20,t=10,b=10))
        st.plotly_chart(fig_hbar, use_container_width=True)

    st.markdown("---")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### 📐 Distribusi IPK Mahasiswa")
        bins       = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.01]
        labels_bin = ['0.0–0.5','0.6–1.0','1.1–1.5','1.6–2.0','2.1–2.5','2.6–3.0','3.1–3.5','3.6–4.0']
        counts     = pd.cut(df_res['IPK'], bins=bins, right=True, labels=labels_bin) \
                       .value_counts().reindex(labels_bin, fill_value=0)
        bin_mids   = np.array([0.25,0.75,1.25,1.75,2.25,2.75,3.25,3.75])
        try:
            from scipy.stats import gaussian_kde
            kde_vals = gaussian_kde(df_res['IPK'].dropna(), bw_method=0.4)(bin_mids) * len(df_res) * 0.5
            has_kde  = True
        except Exception:
            has_kde = False

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(x=labels_bin, y=counts.values,
                                  marker_color='#4fc3f7', marker_line_color='white', marker_line_width=1.5,
                                  text=counts.values, textposition='inside',
                                  textfont=dict(color='white',size=13,family='Syne'), name='Jumlah'))
        if has_kde:
            fig_hist.add_trace(go.Scatter(x=labels_bin, y=kde_vals, mode='lines',
                                          line=dict(color='#e91e8c',width=3,shape='spline'),
                                          name='Distribusi', hoverinfo='skip'))
        fig_hist.update_layout(height=340, xaxis_title='Rentang IPK', yaxis_title='Jumlah Mahasiswa',
                               plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)',
                               showlegend=False, margin=dict(l=10,r=10,t=10,b=10), bargap=0.05)
        st.plotly_chart(fig_hist, use_container_width=True)

    with c4:
        st.markdown("#### 🏆 Rata-rata IPK per Kategori")
        fig_cat = go.Figure(go.Bar(
            x=['✅ Lulus\nTepat Waktu','❌ Tidak\nTepat Waktu'],
            y=[ipk_lulus, ipk_tidak], marker_color=['#667eea','#4facfe'],
            text=[f"{ipk_lulus:.2f}", f"{ipk_tidak:.2f}"],
            textposition='outside', textfont=dict(size=16,family='Syne',color=['#667eea','#4facfe']),
            width=0.4))
        fig_cat.add_hline(y=2.0, line_dash="dot", line_color="#f5576c", line_width=2,
                          annotation_text="Min 2.0", annotation_position="right")
        fig_cat.add_hline(y=3.0, line_dash="dash", line_color="#06d6a0", line_width=1.5,
                          annotation_text="Ideal 3.0", annotation_position="right")
        fig_cat.update_layout(height=340, yaxis=dict(range=[0,4.5],title='Rata-rata IPK'),
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              margin=dict(l=10,r=60,t=10,b=10), showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)


def page_about(model_ok, accuracy, best_params):
    st.markdown("## ℹ️ Tentang Sistem")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="about-card" style="min-height:370px;box-sizing:border-box;">
            <h3>🎯 Tujuan Sistem</h3>
            <p>Sistem prediksi berbasis machine learning untuk menganalisis peluang mahasiswa lulus tepat waktu menggunakan Random Forest + GridSearchCV.</p>
            <ul style="margin-top:.75rem;padding-left:1rem;">
                <li>Identifikasi risiko keterlambatan studi sejak dini</li>
                <li>Rekomendasi tindakan konkret per mahasiswa</li>
                <li>Analisis massal via Upload CSV</li>
            </ul></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="about-card" style="min-height:370px;box-sizing:border-box;">
            <h3>🧠 Cara Kerja</h3>
            <p><b>Step 1</b> — Input data pribadi & IPS Semester 1–6</p>
            <p><b>Step 2</b> — Label Encoding variabel kategorikal</p>
            <p><b>Step 3</b> — Standard Scaler normalisasi fitur</p>
            <p><b>Step 4</b> — GridSearchCV 5-Fold CV → best params</p>
            <p><b>Step 5</b> — Random Forest best estimator → probabilitas</p>
            <p><b>Step 6</b> — Threshold → kelas prediksi + prediction score</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("""<div class="about-card"><h3>📊 Kategori Hasil</h3>
            <p><b>✅ Lulus Tepat Waktu</b><br>Model memprediksi mahasiswa lulus sesuai target waktu studi.</p><br>
            <p><b>❌ Tidak Lulus Tepat Waktu</b><br>Model mendeteksi faktor risiko yang perlu ditangani.</p>
            </div>""", unsafe_allow_html=True)


# SESSION STATE INIT
def init_state():
    defaults = {
        'mode':             'personal',
        'personal_menu':    '📝 Input Data',
        'csv_menu':         '📤 Upload & Prediksi',
        'prediction_done':  False,
        'result':           {},
        'csv_results':      None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# MAIN
def main():
    init_state()
    model, scaler, le_prodi, le_gender, le_status, accuracy, total_data, model_err, best_params = train_model()
    model_ok = model_err is None

    with st.sidebar:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:1rem;">
            <div class="logo-icon-box">🎓</div>
            <div>
                <div class="sb-logo-title">Graduation<br>Prediction<br>System</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown('<span class="sb-section">System Status</span>', unsafe_allow_html=True)
        if model_ok:
            st.markdown('<div class="status-badge"><div class="status-dot"></div>'
                        '<span class="status-text">Model Ready</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge" style="background:rgba(251,191,36,0.1);'
                        'border-color:rgba(251,191,36,0.28);"><div class="status-dot" '
                        'style="background:#fbbf24;box-shadow:0 0 7px #fbbf24;"></div>'
                        '<span style="font-size:.78rem;color:#fbbf24 !important;font-weight:600;">'
                        'Demo Mode</span></div>', unsafe_allow_html=True)

        st.markdown("---")

        st.markdown('<span class="sb-section">Mode Input</span>', unsafe_allow_html=True)
        st.markdown('<span class="sb-mode-hint">▼ pilih mode di sini</span>', unsafe_allow_html=True)

        def _on_mode():
            sel = st.session_state['_mode_select']
            st.session_state.mode = 'personal' if 'Personal' in sel else 'csv'
            if st.session_state.mode == 'personal':
                st.session_state.personal_menu = '📝 Input Data'
            else:
                st.session_state.csv_menu = '📤 Upload & Prediksi'

        mode_opts = ['👤 Input Data Personal', '📂 Upload CSV']
        mode_idx  = 0 if st.session_state.mode == 'personal' else 1
        st.selectbox("Mode", mode_opts, index=mode_idx,
                     key="_mode_select", label_visibility="collapsed",
                     on_change=_on_mode)

        st.markdown("---")

        st.markdown('<span class="sb-section">Navigasi</span>', unsafe_allow_html=True)

        if st.session_state.mode == 'personal':
            nav_opts = ['📝 Input Data','📊 Hasil Prediksi','🔍 Analisis',
                        '💡 Rekomendasi','📈 Visualisasi','ℹ️ About']
            cur_menu = st.session_state.personal_menu
        else:
            nav_opts = ['📤 Upload & Prediksi','📊 Hasil Prediksi','📈 Visualisasi','ℹ️ About']
            cur_menu = st.session_state.csv_menu

        nav_idx = nav_opts.index(cur_menu) if cur_menu in nav_opts else 0

        def _on_nav():
            sel = st.session_state['_nav_radio']
            if st.session_state.mode == 'personal':
                st.session_state.personal_menu = sel
            else:
                st.session_state.csv_menu = sel

        st.radio("Nav", nav_opts, index=nav_idx,
                 key="_nav_radio", label_visibility="collapsed",
                 on_change=_on_nav)

    render_header(accuracy, total_data, model_ok, best_params)

    if st.session_state.mode == 'personal':
        menu = st.session_state.personal_menu
        if   menu == '📝 Input Data':      page_input(model, scaler, le_prodi, le_gender, le_status, model_ok)
        elif menu == '📊 Hasil Prediksi':  page_hasil_personal()
        elif menu == '🔍 Analisis':        page_analisis_personal(model, model_ok)
        elif menu == '💡 Rekomendasi':     page_rekomendasi_personal()
        elif menu == '📈 Visualisasi':     page_visualisasi_personal()
        elif menu == 'ℹ️ About':           page_about(model_ok, accuracy, best_params)
    else:
        menu = st.session_state.csv_menu
        if   menu == '📤 Upload & Prediksi': page_upload_csv(model, scaler, le_prodi, le_gender, le_status, model_ok)
        elif menu == '📊 Hasil Prediksi':    page_hasil_csv()
        elif menu == '📈 Visualisasi':       page_visualisasi_csv()
        elif menu == 'ℹ️ About':             page_about(model_ok, accuracy, best_params)

    st.markdown("---")
    st.markdown("""<div style="text-align:center;padding:.75rem;color:#7c6fa0;font-size:.8rem;">
        <strong>🎓 Sistem Prediksi Kelulusan Mahasiswa</strong> · BINUS University · Data Science · © 2025<br>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
