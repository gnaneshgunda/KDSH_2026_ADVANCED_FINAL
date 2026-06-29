"""
Narrative Consistency RAG Auditor — Premium Streamlit Dashboard
A showcase-grade UI for verifying character backstories against source novels.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import time
import json
import re
import logging
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

# ─────────────────────── PAGE CONFIG ───────────────────────
st.set_page_config(
    page_title="Narrative RAG Auditor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────── PREMIUM CSS THEME ───────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: #06080d;
        color: #e2e8f0;
    }

    .stApp {
        background: linear-gradient(180deg, #06080d 0%, #0a0e1a 30%, #080c15 100%);
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

    /* ── Header Hero ── */
    .hero-container {
        position: relative;
        padding: 2rem 2.5rem;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.6) 0%, rgba(30, 41, 59, 0.3) 50%, rgba(15, 23, 42, 0.6) 100%);
        border-radius: 16px;
        border: 1px solid rgba(56, 189, 248, 0.12);
        margin-bottom: 1.5rem;
        overflow: hidden;
        box-shadow: 0 0 60px rgba(56, 189, 248, 0.04), 0 1px 3px rgba(0,0,0,0.3);
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #38bdf8, #818cf8, #c084fc, transparent);
        animation: shimmer 4s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 40%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.7px;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        font-weight: 400;
        margin: 0;
        letter-spacing: 0.2px;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 0.75rem;
    }
    .badge-rag {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
    }
    .badge-nvidia {
        background: rgba(118, 185, 0, 0.1);
        border: 1px solid rgba(118, 185, 0, 0.3);
        color: #76b900;
        margin-left: 8px;
    }

    /* ── Glass Cards ── */
    .glass-card {
        background: rgba(15, 23, 42, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(51, 65, 85, 0.35);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.2);
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.06);
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-header .icon {
        font-size: 1.2rem;
    }

    /* ── Verdict Banners ── */
    .verdict-panel {
        padding: 1.5rem 2rem;
        border-radius: 14px;
        text-align: center;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    .verdict-panel::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        border-radius: 14px;
        opacity: 0.06;
    }
    .verdict-consistent {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(5, 150, 105, 0.12));
        border: 1px solid rgba(16, 185, 129, 0.35);
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.06);
    }
    .verdict-inconsistent {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.08), rgba(225, 29, 72, 0.12));
        border: 1px solid rgba(244, 63, 94, 0.35);
        box-shadow: 0 0 30px rgba(244, 63, 94, 0.06);
    }
    .verdict-unknown {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.08), rgba(217, 119, 6, 0.12));
        border: 1px solid rgba(245, 158, 11, 0.35);
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.06);
    }
    .verdict-icon { font-size: 2rem; margin-bottom: 0.3rem; }
    .verdict-label {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .verdict-desc {
        font-size: 0.85rem;
        opacity: 0.8;
        margin-top: 0.3rem;
        font-weight: 400;
    }
    .verdict-consistent .verdict-label { color: #34d399; }
    .verdict-inconsistent .verdict-label { color: #fb7185; }
    .verdict-unknown .verdict-label { color: #fbbf24; }
    .verdict-consistent .verdict-desc { color: #6ee7b7; }
    .verdict-inconsistent .verdict-desc { color: #fda4af; }
    .verdict-unknown .verdict-desc { color: #fde68a; }

    /* ── Claim Badges ── */
    .claim-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-supported {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-contradicted {
        background: rgba(244, 63, 94, 0.12);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }
    .badge-not-mentioned {
        background: rgba(100, 116, 139, 0.15);
        color: #94a3b8;
        border: 1px solid rgba(100, 116, 139, 0.3);
    }

    /* ── Evidence Passage ── */
    .evidence-block {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(51, 65, 85, 0.3);
        border-left: 3px solid rgba(56, 189, 248, 0.4);
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.85rem;
        line-height: 1.6;
        color: #cbd5e1;
    }
    .evidence-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: #475569;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    /* ── Highlight ── */
    .hl-char {
        background: rgba(56, 189, 248, 0.15);
        color: #7dd3fc;
        padding: 1px 5px;
        border-radius: 3px;
        border: 1px solid rgba(56, 189, 248, 0.2);
        font-weight: 500;
    }

    /* ── Metric Cards ── */
    .metric-row {
        display: flex;
        gap: 12px;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(51, 65, 85, 0.3);
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
        font-weight: 600;
    }
    .metric-cyan .metric-value { color: #38bdf8; }
    .metric-green .metric-value { color: #34d399; }
    .metric-red .metric-value { color: #fb7185; }
    .metric-amber .metric-value { color: #fbbf24; }
    .metric-purple .metric-value { color: #c084fc; }

    /* ── Trace Timeline ── */
    .trace-step {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        padding: 10px 0;
        border-bottom: 1px solid rgba(51, 65, 85, 0.2);
    }
    .trace-step:last-child { border-bottom: none; }
    .trace-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-top: 5px;
        flex-shrink: 0;
    }
    .trace-dot-blue { background: #38bdf8; box-shadow: 0 0 8px rgba(56, 189, 248, 0.4); }
    .trace-dot-green { background: #34d399; box-shadow: 0 0 8px rgba(52, 211, 153, 0.4); }
    .trace-dot-red { background: #fb7185; box-shadow: 0 0 8px rgba(251, 113, 133, 0.4); }
    .trace-dot-amber { background: #fbbf24; box-shadow: 0 0 8px rgba(251, 191, 36, 0.4); }
    .trace-content { flex: 1; }
    .trace-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .trace-detail {
        font-size: 0.75rem;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
    }
    .trace-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #475569;
        white-space: nowrap;
    }

    /* ── Tab Overrides ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(15, 23, 42, 0.6);
        padding: 5px 8px;
        border-radius: 10px;
        border: 1px solid rgba(51, 65, 85, 0.25);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 22px;
        background: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.85rem;
        border: none;
        transition: all 0.25s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #cbd5e1;
        background: rgba(255,255,255, 0.03);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.1) !important;
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #070a12 !important;
        border-right: 1px solid rgba(51, 65, 85, 0.25);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1rem;
        color: #94a3b8;
    }

    /* ── Buttons ── */
    .stButton>button {
        background: linear-gradient(135deg, #0369a1 0%, #4338ca 100%);
        border: none;
        color: white;
        font-weight: 700;
        border-radius: 10px;
        padding: 0.65rem 1.8rem;
        font-size: 0.85rem;
        box-shadow: 0 4px 14px rgba(3, 105, 161, 0.25);
        transition: all 0.2s ease;
        letter-spacing: 0.3px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(3, 105, 161, 0.35);
        background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%);
        color: white;
    }
    .stButton>button:active { transform: translateY(0); }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(51, 65, 85, 0.3) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }

    /* ── Data Table ── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* ── Pipeline Architecture ── */
    .arch-flow {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        padding: 1.5rem 0;
        flex-wrap: wrap;
    }
    .arch-node {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(51, 65, 85, 0.4);
        border-radius: 10px;
        padding: 12px 18px;
        text-align: center;
        min-width: 110px;
        transition: all 0.2s ease;
    }
    .arch-node:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.08);
    }
    .arch-node-icon { font-size: 1.4rem; margin-bottom: 4px; }
    .arch-node-label {
        font-size: 0.7rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .arch-arrow {
        color: #334155;
        font-size: 1.2rem;
        padding: 0 6px;
    }

    /* ── Book Card ── */
    .book-card {
        background: rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(51, 65, 85, 0.3);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .book-title {
        font-weight: 700;
        color: #7dd3fc;
        font-size: 0.95rem;
    }
    .book-meta {
        font-size: 0.75rem;
        color: #475569;
        font-family: 'JetBrains Mono', monospace;
    }
    .book-status {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 0.7rem;
        color: #34d399;
        font-weight: 600;
    }

    /* ── About Section ── */
    .about-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.5), rgba(30, 41, 59, 0.3));
        border: 1px solid rgba(51, 65, 85, 0.3);
        border-radius: 14px;
        padding: 2rem;
        margin-bottom: 1rem;
    }
    .about-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #f1f5f9;
        margin-bottom: 0.75rem;
    }
    .about-text {
        color: #94a3b8;
        font-size: 0.88rem;
        line-height: 1.7;
    }
    .step-list {
        list-style: none;
        padding: 0;
        counter-reset: step-counter;
    }
    .step-list li {
        counter-increment: step-counter;
        padding: 10px 0 10px 48px;
        position: relative;
        color: #94a3b8;
        font-size: 0.88rem;
        border-bottom: 1px solid rgba(51, 65, 85, 0.2);
    }
    .step-list li:last-child { border-bottom: none; }
    .step-list li::before {
        content: counter(step-counter);
        position: absolute;
        left: 0;
        top: 8px;
        width: 30px;
        height: 30px;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.75rem;
        color: #38bdf8;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────── PIPELINE INIT ───────────────────────

@st.cache_resource
def get_rag_instance():
    """Get or initialize the pipeline logic class"""
    from pipeline import AdvancedNarrativeConsistencyRAG
    return AdvancedNarrativeConsistencyRAG()

try:
    rag = get_rag_instance()
    cached_books = rag.index_manager.list_cached_books()
    corpus = rag.index_manager.load_or_build()
except Exception as e:
    st.error(f"Error loading RAG pipeline: {e}")
    st.stop()

@st.cache_data
def load_datasets():
    train_path = Path("db/train.csv")
    test_path = Path("db/test.csv")
    train_df = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()
    test_df = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
    return train_df, test_df

train_df, test_df = load_datasets()

# Build character lists per book
chars_per_book = {}
for df in [train_df, test_df]:
    if not df.empty:
        for _, row in df.iterrows():
            b = str(row.get("book_name", "")).strip().lower()
            c = str(row.get("char", "")).strip()
            if b and c:
                chars_per_book.setdefault(b, set()).add(c)
chars_per_book = {b: sorted(list(c)) for b, c in chars_per_book.items()}


# ─────────────────────── SIDEBAR ───────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size: 2rem;">🔬</div>
        <div style="font-size: 0.95rem; font-weight: 700; color: #e2e8f0; letter-spacing: -0.3px;">RAG Auditor</div>
        <div style="font-size: 0.7rem; color: #475569; font-weight: 500;">v2.0 — Pipeline Controls</div>
    </div>
    """, unsafe_allow_html=True)

    # Connection status
    has_api_key = bool(os.getenv("NVIDIA_API_KEY")) or st.session_state.get("api_key")
    if has_api_key:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); 
             border-radius: 8px; padding: 8px 12px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
            <div style="width: 7px; height: 7px; border-radius: 50%; background: #34d399; 
                 box-shadow: 0 0 6px rgba(52, 211, 153, 0.5);"></div>
            <span style="font-size: 0.75rem; color: #34d399; font-weight: 600;">NVIDIA NIM Connected</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); 
             border-radius: 8px; padding: 8px 12px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
            <div style="width: 7px; height: 7px; border-radius: 50%; background: #fbbf24; 
                 box-shadow: 0 0 6px rgba(251, 191, 36, 0.5);"></div>
            <span style="font-size: 0.75rem; color: #fbbf24; font-weight: 600;">Local Fallback Mode</span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔑 API Credentials", expanded=not has_api_key):
        input_key = st.text_input(
            "NVIDIA API Key", type="password",
            value=os.getenv("NVIDIA_API_KEY", st.session_state.get("api_key", ""))
        )
        if input_key:
            st.session_state["api_key"] = input_key
            rag.client.api_key = input_key
            rag.client.use_nvidia_api = True
            rag.client.use_hf = False
            rag.client.active_embedding_dim = 2048
        elif "api_key" in st.session_state:
            del st.session_state["api_key"]
            rag.client.api_key = ""
            rag.client.use_nvidia_api = False
            rag.client.use_hf = True
            rag.client.active_embedding_dim = 384

    st.markdown("---")
    st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#94a3b8; margin-bottom:8px;">⚙️ RETRIEVAL SETTINGS</div>', unsafe_allow_html=True)

    use_rerank = st.toggle("NVIDIA Reranking", value=True,
                           help="Use nv-rerank-qa-mixtral-8x7b for passage reranking")
    enable_fallback = st.toggle("Retrieval Fallback Loop", value=True,
                                help="If inconclusive, expand search and re-verify")
    enable_negation = st.toggle("Negation Verification", value=False,
                                help="Double-check by querying for negated claims")

    top_k = st.slider("Retrieval Top-K", 1, 10, 5, 1)
    context_window = st.slider("Context Window (±N)", 0, 2, 1, 1,
                               help="Include ±N neighboring chunks for narrative flow")

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size: 0.7rem; color: #334155; text-align: center; padding-top: 0.5rem;">
        {len(corpus)} books indexed · {sum(len(v) for v in corpus.values())} chunks loaded
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────── HERO HEADER ───────────────────────

st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Narrative Consistency RAG Auditor</h1>
    <p class="hero-subtitle">Advanced fact-checking system that decomposes character backstories into atomic claims 
    and verifies each against source novel passages using multi-stage retrieval-augmented generation.</p>
    <div>
        <span class="hero-badge badge-rag">⚡ RAG Pipeline</span>
        <span class="hero-badge badge-nvidia">🟢 NVIDIA NIM</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Pipeline architecture mini-diagram
st.markdown("""
<div class="arch-flow">
    <div class="arch-node"><div class="arch-node-icon">📝</div><div class="arch-node-label">Backstory</div></div>
    <div class="arch-arrow">→</div>
    <div class="arch-node"><div class="arch-node-icon">🔪</div><div class="arch-node-label">Claim Extract</div></div>
    <div class="arch-arrow">→</div>
    <div class="arch-node"><div class="arch-node-icon">🔍</div><div class="arch-node-label">Hybrid Retrieve</div></div>
    <div class="arch-arrow">→</div>
    <div class="arch-node"><div class="arch-node-icon">📊</div><div class="arch-node-label">Rerank</div></div>
    <div class="arch-arrow">→</div>
    <div class="arch-node"><div class="arch-node-icon">✅</div><div class="arch-node-label">Verify Claims</div></div>
    <div class="arch-arrow">→</div>
    <div class="arch-node"><div class="arch-node-icon">⚖️</div><div class="arch-node-label">Verdict</div></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────── TABS ───────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Consistency Playground",
    "📊 Batch Evaluation",
    "📚 Index Manager",
    "ℹ️ Architecture & About"
])


# ═══════════════════════ TAB 1: PLAYGROUND ═══════════════════════
with tab1:
    col_input, col_spacer, col_result = st.columns([4, 0.3, 6])

    with col_input:
        st.markdown('<div class="section-header"><span class="icon">📋</span> Test Case Setup</div>', unsafe_allow_html=True)

        # Book selection
        available_book_keys = sorted(list(corpus.keys()))
        selected_book_key = st.selectbox(
            "Target Book", options=available_book_keys,
            format_func=lambda x: x.replace("_", " ").title()
        )

        # Character selection
        book_chars = chars_per_book.get(selected_book_key, ["Generic Character"])
        selected_char = st.selectbox("Character", options=book_chars)

        # Dataset sample loader
        matching_rows = pd.DataFrame()
        if not train_df.empty:
            matching_rows = train_df[
                (train_df["book_name"].str.lower() == selected_book_key) &
                (train_df["char"].str.lower() == selected_char.lower())
            ]

        sample_choices = ["✍️ Custom Backstory"]
        if not matching_rows.empty:
            for _, r in matching_rows.iterrows():
                lbl_icon = "✅" if str(r.get("label", "")).lower() == "consistent" else "❌"
                sample_choices.append(f"{lbl_icon} Sample #{r['id']} ({r['label']})")

        selected_sample = st.selectbox("Load Example (Optional)", options=sample_choices)

        # Populate backstory
        initial_backstory = ""
        ground_truth_lbl = None
        if selected_sample != "✍️ Custom Backstory":
            sample_id = int(selected_sample.split("#")[1].split(" ")[0])
            sample_row = train_df[train_df["id"] == sample_id].iloc[0]
            initial_backstory = str(sample_row.get("content", ""))
            if not initial_backstory or initial_backstory == "nan":
                initial_backstory = str(sample_row.get("caption", ""))
            ground_truth_lbl = sample_row.get("label", None)

        backstory_text = st.text_area(
            "Character Backstory", value=initial_backstory, height=200,
            placeholder="Paste or type the backstory to verify against the source novel..."
        )

        if ground_truth_lbl:
            lbl_color = "#34d399" if str(ground_truth_lbl).lower() == "consistent" else "#fb7185"
            st.markdown(f"""
            <div style="background: rgba(56, 189, 248, 0.06); border: 1px solid rgba(56, 189, 248, 0.15); 
                 border-radius: 8px; padding: 8px 14px; margin: 8px 0;">
                <span style="font-size: 0.75rem; color: #64748b;">Ground Truth:</span>
                <span style="font-size: 0.85rem; font-weight: 700; color: {lbl_color}; margin-left: 6px;">
                    {str(ground_truth_lbl).upper()}
                </span>
            </div>
            """, unsafe_allow_html=True)

        verify_btn = st.button("🔬 Run Consistency Audit", use_container_width=True)

    with col_result:
        if verify_btn:
            if not backstory_text.strip():
                st.warning("Please provide backstory text to analyze.")
            elif selected_book_key not in corpus:
                st.error(f"Book '{selected_book_key}' is not indexed.")
            else:
                with st.spinner(""):
                    # Show animated analysis status
                    status_placeholder = st.empty()
                    status_placeholder.markdown("""
                    <div class="glass-card" style="text-align:center; padding: 2rem;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔬</div>
                        <div style="color: #38bdf8; font-weight: 600; font-size: 0.9rem;">Analyzing Narrative Consistency...</div>
                        <div style="color: #475569; font-size: 0.75rem; margin-top: 4px;">
                            Extracting claims → Retrieving evidence → Verifying facts
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    start_time = time.time()
                    analysis = rag.analyze_backstory(
                        book_name=selected_book_key,
                        character=selected_char,
                        backstory_text=backstory_text,
                        narrative_chunks=corpus[selected_book_key],
                        use_rerank=use_rerank,
                        top_k=top_k,
                        context_window=context_window,
                        enable_fallback=enable_fallback,
                        enable_negation=enable_negation
                    )
                    elapsed = time.time() - start_time
                    status_placeholder.empty()

                # ── Verdict Banner ──
                v_val = analysis.verdict
                conf = analysis.confidence

                if v_val == "1":
                    if conf >= 0.75:
                        v_icon, v_label, v_desc, v_class = "✅", "CONSISTENT", "Claims align with source narrative evidence.", "verdict-consistent"
                    else:
                        v_icon, v_label, v_desc, v_class = "🟡", "LIKELY CONSISTENT", "No contradictions found, but evidence is largely implicit.", "verdict-unknown"
                elif v_val == "0":
                    v_icon, v_label, v_desc, v_class = "❌", "INCONSISTENT", "Direct contradiction detected between claims and source text.", "verdict-inconsistent"
                else:
                    v_icon, v_label, v_desc, v_class = "❔", "INSUFFICIENT EVIDENCE", "Not enough narrative context to determine consistency.", "verdict-unknown"

                st.markdown(f"""
                <div class="verdict-panel {v_class}">
                    <div class="verdict-icon">{v_icon}</div>
                    <div class="verdict-label">{v_label}</div>
                    <div class="verdict-desc">{v_desc}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Metrics Row ──
                n_claims = len(analysis.claim_results)
                n_supported = analysis.num_supported_claims
                n_contradicted = analysis.num_contradicted_claims
                n_unknown = analysis.num_unknown_claims

                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card metric-cyan">
                        <div class="metric-value">{conf*100:.0f}%</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                    <div class="metric-card metric-purple">
                        <div class="metric-value">{n_claims}</div>
                        <div class="metric-label">Claims</div>
                    </div>
                    <div class="metric-card metric-green">
                        <div class="metric-value">{n_supported}</div>
                        <div class="metric-label">Supported</div>
                    </div>
                    <div class="metric-card metric-red">
                        <div class="metric-value">{n_contradicted}</div>
                        <div class="metric-label">Contradicted</div>
                    </div>
                    <div class="metric-card metric-amber">
                        <div class="metric-value">{elapsed:.1f}s</div>
                        <div class="metric-label">Latency</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Confidence Gauge ──
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=conf * 100,
                    number={'suffix': '%', 'font': {'size': 28, 'color': '#e2e8f0', 'family': 'JetBrains Mono'}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Auditor Confidence", 'font': {'size': 13, 'color': '#64748b', 'family': 'Inter'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#334155', 'dtick': 25},
                        'bar': {'color': '#38bdf8', 'thickness': 0.3},
                        'bgcolor': '#0f172a',
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 40], 'color': 'rgba(244, 63, 94, 0.08)'},
                            {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.08)'},
                            {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.08)'}
                        ],
                        'threshold': {
                            'line': {'color': '#f1f5f9', 'width': 2},
                            'thickness': 0.8,
                            'value': conf * 100
                        }
                    }
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#e2e8f0', 'family': 'Inter'},
                    height=180, margin=dict(l=30, r=30, t=50, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── Rationale ──
                st.markdown('<div class="section-header"><span class="icon">💡</span> Audit Rationale</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="glass-card" style="border-left: 3px solid {'#34d399' if v_val == '1' else '#fb7185' if v_val == '0' else '#fbbf24'};">
                    <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.7;">{analysis.rationale}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Pipeline Trace (if available) ──
                trace = getattr(analysis, 'trace', {})
                if trace and trace.get('per_claim_traces'):
                    st.markdown('<div class="section-header"><span class="icon">📡</span> Pipeline Trace</div>', unsafe_allow_html=True)
                    with st.expander("View execution trace", expanded=False):
                        for ct in trace['per_claim_traces']:
                            verd = ct.get('verification_verdict', 'UNKNOWN')
                            dot_class = 'trace-dot-green' if verd == 'SUPPORTED' else 'trace-dot-red' if verd == 'CONTRADICTED' else 'trace-dot-amber'
                            st.markdown(f"""
                            <div class="trace-step">
                                <div class="trace-dot {dot_class}"></div>
                                <div class="trace-content">
                                    <div class="trace-title">{ct.get('claim', '')[:80]}...</div>
                                    <div class="trace-detail">
                                        {ct.get('chunks_retrieved', '?')} chunks · 
                                        {ct.get('retrieval_rounds', 1)} round(s) · 
                                        verdict: {verd}
                                    </div>
                                </div>
                                <div class="trace-time">{ct.get('time_ms', 0):.0f}ms</div>
                            </div>
                            """, unsafe_allow_html=True)

                # ── Claim Details ──
                st.markdown('<div class="section-header"><span class="icon">🧩</span> Atomic Claim Verification</div>', unsafe_allow_html=True)

                for idx, c_res in enumerate(analysis.claim_results, 1):
                    verdict_label = c_res.get("verdict", "NOT_MENTIONED")
                    if verdict_label == "SUPPORTED":
                        badge_class = "badge-supported"
                    elif verdict_label == "CONTRADICTED":
                        badge_class = "badge-contradicted"
                    else:
                        badge_class = "badge-not-mentioned"

                    conf_val = c_res.get("confidence", 0.5)
                    claim_preview = c_res.get("claim", "")[:65]

                    with st.expander(f"Claim #{idx} — {claim_preview}..."):
                        st.markdown(f"""
                        <div style="margin-bottom: 10px;">
                            <span class="claim-badge {badge_class}">{verdict_label}</span>
                            <span style="font-family: 'JetBrains Mono'; font-size: 0.7rem; color: #475569; margin-left: 8px;">
                                conf: {conf_val:.2f}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"**Claim:** {c_res.get('claim', '')}")
                        st.markdown(f"**Reasoning:** {c_res.get('rationale', 'N/A')}")

                        if c_res.get("evidence"):
                            st.markdown("**Retrieved Evidence:**")
                            for j, chunk in enumerate(c_res["evidence"], 1):
                                highlighted = chunk
                                if selected_char:
                                    pattern = re.compile(re.escape(selected_char), re.IGNORECASE)
                                    highlighted = pattern.sub(
                                        f'<span class="hl-char">{selected_char}</span>', chunk
                                    )
                                st.markdown(f"""
                                <div class="evidence-block">
                                    <div class="evidence-label">Passage {j}</div>
                                    {highlighted}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.caption("No matching passages retrieved.")

                # ── Export Button ──
                export_data = {
                    "book": selected_book_key,
                    "character": selected_char,
                    "verdict": v_label,
                    "confidence": conf,
                    "elapsed_seconds": elapsed,
                    "num_claims": n_claims,
                    "num_supported": n_supported,
                    "num_contradicted": n_contradicted,
                    "rationale": analysis.rationale,
                    "claims": analysis.claim_results
                }
                st.download_button(
                    "📥 Export Analysis (JSON)",
                    data=json.dumps(export_data, indent=2, default=str),
                    file_name=f"audit_{selected_book_key}_{selected_char}.json",
                    mime="application/json"
                )
        else:
            # Empty state
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 3rem 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.4;">🔬</div>
                <div style="font-size: 1rem; font-weight: 600; color: #475569; margin-bottom: 0.5rem;">
                    Ready for Analysis
                </div>
                <div style="font-size: 0.82rem; color: #334155; max-width: 400px; margin: 0 auto; line-height: 1.6;">
                    Select a book and character, paste a backstory, then click 
                    <strong style="color:#38bdf8;">Run Consistency Audit</strong> 
                    to verify narrative alignment.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════ TAB 2: BATCH EVALUATION ═══════════════════════
with tab2:
    st.markdown('<div class="section-header"><span class="icon">📊</span> Benchmark Evaluation Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color: #64748b; font-size: 0.85rem; margin-bottom: 1rem; line-height: 1.5;">
        Run batch verification over the benchmark datasets to measure accuracy, precision, recall, F1, 
        and trace error patterns across books and characters.
    </div>
    """, unsafe_allow_html=True)

    bench_col1, bench_col2 = st.columns([3, 7])

    with bench_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="icon">⚙️</span> Config</div>', unsafe_allow_html=True)

        bench_dataset = st.radio("Dataset", ["Train (labeled)", "Test (labeled)"])
        bench_limit = st.slider("Row limit", 5, 80, 15, 5,
                                help="Limit rows to avoid API rate limits")
        run_bench_btn = st.button("▶ Run Evaluation", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with bench_col2:
        if run_bench_btn:
            eval_df = train_df if "Train" in bench_dataset else test_df
            if eval_df.empty:
                st.error("Dataset is empty or not found.")
            else:
                prog_bar = st.progress(0.0)
                status_txt = st.empty()
                rows = eval_df.head(bench_limit)
                results = []
                total = len(rows)

                for idx, (_, row) in enumerate(rows.iterrows()):
                    r_id = row.get("id")
                    b_name = str(row.get("book_name", "")).strip().lower()
                    char_name = str(row.get("char", "")).strip()
                    backstory_c = str(row.get("content", ""))
                    if not backstory_c or backstory_c == "nan":
                        backstory_c = str(row.get("caption", ""))
                    true_label = str(row.get("label", "")).strip().lower()

                    status_txt.markdown(f"""
                    <div style="font-size: 0.8rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">
                        [{idx+1}/{total}] Processing #{r_id} · {b_name} × {char_name}
                    </div>
                    """, unsafe_allow_html=True)
                    prog_bar.progress((idx + 1) / total)

                    if b_name not in corpus:
                        results.append({
                            "id": r_id, "char": char_name, "book": b_name,
                            "true": true_label, "pred_lbl": "unknown",
                            "correct": False, "conf": 0.0, "reason": "Book not indexed"
                        })
                        continue

                    try:
                        analysis = rag.analyze_backstory(
                            book_name=b_name, character=char_name,
                            backstory_text=backstory_c,
                            narrative_chunks=corpus[b_name],
                            use_rerank=use_rerank, top_k=top_k,
                            context_window=context_window,
                            enable_fallback=enable_fallback,
                            enable_negation=enable_negation
                        )
                        pred_lbl = "contradict" if analysis.verdict == "0" else "consistent"
                        results.append({
                            "id": r_id, "char": char_name, "book": b_name,
                            "true": true_label, "pred_lbl": pred_lbl,
                            "correct": (pred_lbl == true_label),
                            "conf": analysis.confidence,
                            "reason": analysis.rationale[:200]
                        })
                    except Exception as ex:
                        results.append({
                            "id": r_id, "char": char_name, "book": b_name,
                            "true": true_label, "pred_lbl": "error",
                            "correct": False, "conf": 0.0, "reason": str(ex)[:200]
                        })

                status_txt.empty()
                prog_bar.empty()

                res_df = pd.DataFrame(results)

                # ── Compute Metrics ──
                tp = len(res_df[(res_df["true"] == "consistent") & (res_df["pred_lbl"] == "consistent")])
                tn = len(res_df[(res_df["true"] == "contradict") & (res_df["pred_lbl"] == "contradict")])
                fp = len(res_df[(res_df["true"] == "contradict") & (res_df["pred_lbl"] == "consistent")])
                fn = len(res_df[(res_df["true"] == "consistent") & (res_df["pred_lbl"] == "contradict")])

                accuracy = (tp + tn) / len(res_df) if len(res_df) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                # ── Metric Cards ──
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card metric-cyan">
                        <div class="metric-value">{accuracy*100:.1f}%</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    <div class="metric-card metric-green">
                        <div class="metric-value">{precision*100:.1f}%</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    <div class="metric-card metric-amber">
                        <div class="metric-value">{recall*100:.1f}%</div>
                        <div class="metric-label">Recall</div>
                    </div>
                    <div class="metric-card metric-purple">
                        <div class="metric-value">{f1*100:.1f}%</div>
                        <div class="metric-label">F1 Score</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Confusion Matrix ──
                cm_col, dist_col = st.columns(2)

                with cm_col:
                    st.markdown("##### Confusion Matrix")
                    z_vals = [[tn, fp], [fn, tp]]
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=z_vals,
                        x=['Pred: Contradict', 'Pred: Consistent'],
                        y=['True: Contradict', 'True: Consistent'],
                        text=[[str(tn), str(fp)], [str(fn), str(tp)]],
                        texttemplate="%{text}",
                        textfont={"size": 18, "color": "#f1f5f9", "family": "JetBrains Mono"},
                        colorscale=[[0, '#0f172a'], [0.5, '#1e3a5f'], [1, '#0ea5e9']],
                        showscale=False,
                        hoverinfo='skip'
                    ))
                    fig_cm.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#94a3b8', 'family': 'Inter', 'size': 12},
                        height=260, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(side='bottom'), yaxis=dict(autorange='reversed')
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                with dist_col:
                    st.markdown("##### Confidence Distribution")
                    fig_hist = go.Figure()
                    correct_confs = res_df[res_df["correct"] == True]["conf"].tolist()
                    incorrect_confs = res_df[res_df["correct"] == False]["conf"].tolist()

                    if correct_confs:
                        fig_hist.add_trace(go.Histogram(
                            x=correct_confs, name="Correct",
                            marker_color='rgba(52, 211, 153, 0.6)',
                            nbinsx=10, opacity=0.8
                        ))
                    if incorrect_confs:
                        fig_hist.add_trace(go.Histogram(
                            x=incorrect_confs, name="Incorrect",
                            marker_color='rgba(251, 113, 133, 0.6)',
                            nbinsx=10, opacity=0.8
                        ))
                    fig_hist.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#94a3b8', 'family': 'Inter'},
                        height=260, margin=dict(l=10, r=10, t=10, b=10),
                        barmode='overlay',
                        legend=dict(font=dict(size=10)),
                        xaxis=dict(title="Confidence", gridcolor='rgba(51,65,85,0.2)'),
                        yaxis=dict(title="Count", gridcolor='rgba(51,65,85,0.2)')
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                # ── Error by Book ──
                errors_df = res_df[res_df["correct"] == False]
                if not errors_df.empty:
                    err_col1, err_col2 = st.columns(2)

                    with err_col1:
                        st.markdown("##### Errors by Book")
                        book_errors = errors_df["book"].value_counts()
                        fig_bar = go.Figure(go.Bar(
                            x=book_errors.values,
                            y=[b.replace("_", " ").title() for b in book_errors.index],
                            orientation='h',
                            marker_color='rgba(251, 113, 133, 0.7)',
                            text=book_errors.values,
                            textposition='outside',
                            textfont=dict(color='#fb7185', family='JetBrains Mono', size=12)
                        ))
                        fig_bar.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': '#94a3b8', 'family': 'Inter'},
                            height=200, margin=dict(l=10, r=40, t=10, b=10),
                            xaxis=dict(showgrid=False, showticklabels=False),
                            yaxis=dict(autorange='reversed')
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                    with err_col2:
                        st.markdown("##### Error Type Breakdown")
                        fp_count = len(errors_df[errors_df["pred_lbl"] == "consistent"])
                        fn_count = len(errors_df[errors_df["pred_lbl"] == "contradict"])
                        other_count = len(errors_df) - fp_count - fn_count

                        fig_pie = go.Figure(go.Pie(
                            labels=['False Positive', 'False Negative', 'Other'],
                            values=[fp_count, fn_count, other_count],
                            marker=dict(colors=['rgba(251, 191, 36, 0.8)', 'rgba(251, 113, 133, 0.8)', 'rgba(100, 116, 139, 0.6)']),
                            textinfo='label+value',
                            textfont=dict(size=11, family='Inter'),
                            hole=0.45
                        ))
                        fig_pie.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': '#94a3b8', 'family': 'Inter'},
                            height=200, margin=dict(l=10, r=10, t=10, b=10),
                            showlegend=False
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    # ── Error Details Table ──
                    st.markdown("##### Misclassified Cases")
                    st.dataframe(
                        errors_df[["id", "char", "book", "true", "pred_lbl", "conf", "reason"]],
                        use_container_width=True, hide_index=True,
                        column_config={
                            "id": "ID",
                            "char": "Character",
                            "book": "Book",
                            "true": "Ground Truth",
                            "pred_lbl": "Prediction",
                            "conf": st.column_config.NumberColumn("Confidence", format="%.2f"),
                            "reason": "Reasoning"
                        }
                    )
                else:
                    st.success("🎉 Perfect accuracy on this batch! No misclassifications.")

                # ── Full Results Table ──
                with st.expander("View all results"):
                    st.dataframe(
                        res_df[["id", "char", "book", "true", "pred_lbl", "correct", "conf"]],
                        use_container_width=True, hide_index=True,
                        column_config={
                            "correct": st.column_config.CheckboxColumn("Correct"),
                            "conf": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1)
                        }
                    )
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 3rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.4;">📊</div>
                <div style="font-size: 1rem; font-weight: 600; color: #475569;">
                    Configure & Run Evaluation
                </div>
                <div style="font-size: 0.82rem; color: #334155; max-width: 400px; margin: 0.5rem auto 0; line-height: 1.6;">
                    Select dataset, set row limit, and click <strong style="color: #38bdf8;">Run Evaluation</strong> 
                    to benchmark the pipeline against labeled data.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════ TAB 3: INDEX MANAGER ═══════════════════════
with tab3:
    st.markdown('<div class="section-header"><span class="icon">📚</span> Book Index & Cache Management</div>', unsafe_allow_html=True)

    idx_col1, idx_col2 = st.columns([5, 5])

    with idx_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="icon">💾</span> Cached Indices</div>', unsafe_allow_html=True)

        if cached_books:
            for bk in cached_books:
                ch_count = len(corpus.get(bk, []))
                # Get pkl file size
                pkl_path = Path("db") / f"{bk}.pkl"
                size_mb = pkl_path.stat().st_size / (1024 * 1024) if pkl_path.exists() else 0

                # Entity stats
                chunks = corpus.get(bk, [])
                all_entities = []
                for c in chunks:
                    all_entities.extend(c.entities)
                unique_chars = len(set(all_entities))

                st.markdown(f"""
                <div class="book-card">
                    <div>
                        <div class="book-title">📖 {bk.replace("_", " ").title()}</div>
                        <div class="book-meta">
                            {ch_count} chunks · {size_mb:.1f} MB · {unique_chars} unique entities
                        </div>
                    </div>
                    <div class="book-status">● Loaded</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No cached indices found. Upload a book below.")

        st.markdown('</div>', unsafe_allow_html=True)

        # Chunk Browser
        if corpus:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header"><span class="icon">🔎</span> Chunk Browser</div>', unsafe_allow_html=True)

            browse_book = st.selectbox("Select book to browse", list(corpus.keys()),
                                       format_func=lambda x: x.replace("_", " ").title(),
                                       key="browse_book")
            search_query = st.text_input("Search chunks", placeholder="Type to filter chunks by content...",
                                         key="chunk_search")

            book_chunks = corpus.get(browse_book, [])
            if search_query:
                book_chunks = [c for c in book_chunks if search_query.lower() in c.text.lower()]

            st.caption(f"Showing {min(20, len(book_chunks))} of {len(book_chunks)} chunks")

            for i, chunk in enumerate(book_chunks[:20]):
                with st.expander(f"Chunk {chunk.chunk_id} — {chunk.text[:60]}..."):
                    st.text(chunk.text[:500])
                    if chunk.entities:
                        st.markdown(f"**Entities:** {', '.join(chunk.entities[:8])}")
                    if chunk.temporal_markers:
                        st.markdown(f"**Temporal:** {', '.join(chunk.temporal_markers[:5])}")
                    if chunk.locations:
                        st.markdown(f"**Locations:** {', '.join(chunk.locations[:5])}")

            st.markdown('</div>', unsafe_allow_html=True)

    with idx_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="icon">📥</span> Index New Book</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload novel (.txt)", type=["txt"])

        if uploaded_file is not None:
            new_book_name = st.text_input(
                "Book key (lowercase, underscores)",
                value=uploaded_file.name.replace(".txt", "").strip().lower().replace(" ", "_")
            )

            if st.button("Generate Index & Embeddings", use_container_width=True):
                try:
                    with st.spinner("Building index..."):
                        text_data = uploaded_file.read().decode("utf-8")
                        books_dir = Path(rag.books_dir)
                        books_dir.mkdir(parents=True, exist_ok=True)
                        book_txt_path = books_dir / uploaded_file.name
                        with open(book_txt_path, "w", encoding="utf-8") as bf:
                            bf.write(text_data)

                        pkl_path = Path(rag.db_path) / f"{new_book_name}.pkl"
                        rag.index_manager._build_book_index(new_book_name, book_txt_path, pkl_path)
                        corpus = rag.index_manager.load_or_build()

                    st.success(f"✓ Indexed! {len(corpus.get(new_book_name, []))} chunks created.")
                    time.sleep(1)
                    st.rerun()
                except Exception as ex:
                    st.error(f"Failed to build index: {ex}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Entity Distribution Chart
        if corpus:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header"><span class="icon">👥</span> Entity Distribution</div>', unsafe_allow_html=True)

            entity_data = []
            for book_key, chunks in corpus.items():
                entity_counter = Counter()
                for c in chunks:
                    for e in c.entities:
                        entity_counter[e] += 1
                for ent, count in entity_counter.most_common(8):
                    entity_data.append({
                        "Book": book_key.replace("_", " ").title(),
                        "Entity": ent,
                        "Mentions": count
                    })

            if entity_data:
                ent_df = pd.DataFrame(entity_data)
                fig_ent = px.bar(
                    ent_df, x="Mentions", y="Entity", color="Book",
                    orientation='h',
                    color_discrete_sequence=['#38bdf8', '#c084fc', '#34d399', '#fbbf24']
                )
                fig_ent.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#94a3b8', 'family': 'Inter', 'size': 11},
                    height=300, margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(gridcolor='rgba(51,65,85,0.15)'),
                    yaxis=dict(autorange='reversed'),
                    legend=dict(orientation='h', y=-0.15, font=dict(size=10))
                )
                st.plotly_chart(fig_ent, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════ TAB 4: ARCHITECTURE & ABOUT ═══════════════════════
with tab4:
    about_col1, about_col2 = st.columns([5, 5])

    with about_col1:
        st.markdown("""
        <div class="about-card">
            <div class="about-title">🏗️ System Architecture</div>
            <div class="about-text">
                The Narrative Consistency RAG Auditor uses a multi-stage retrieval-augmented generation pipeline 
                to verify character backstories against source novels. The system decomposes backstories into 
                atomic claims and verifies each independently.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <div class="about-title">📐 Pipeline Stages</div>
            <ol class="step-list">
                <li><strong style="color:#e2e8f0;">Semantic Chunking</strong> — Source novels are split into 
                    paragraph-aware chunks with NLP-based metadata extraction (entities, temporal markers, locations).</li>
                <li><strong style="color:#e2e8f0;">Claim Extraction</strong> — Backstory text is decomposed into 
                    atomic, verifiable claims using LLM-guided extraction with NER pre-processing.</li>
                <li><strong style="color:#e2e8f0;">Hybrid Retrieval</strong> — Each claim triggers multi-query 
                    retrieval combining BM25 keyword matching with semantic cosine similarity, filtered by character metadata.</li>
                <li><strong style="color:#e2e8f0;">Neural Reranking</strong> — Retrieved passages are reranked using 
                    NVIDIA's nv-rerank-qa-mixtral-8x7b model for precision-focused passage selection.</li>
                <li><strong style="color:#e2e8f0;">Chain-of-Thought Verification</strong> — Each claim is verified 
                    against retrieved evidence using structured reasoning with mandatory evidence quoting.</li>
                <li><strong style="color:#e2e8f0;">Weighted Verdict Aggregation</strong> — Individual claim verdicts 
                    are aggregated using category-weighted scoring (identity, temporal, relational claims weighted differently).</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with about_col2:
        st.markdown("""
        <div class="about-card">
            <div class="about-title">⚡ Technical Stack</div>
            <div class="about-text">
                <table style="width:100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid rgba(51,65,85,0.3);">
                        <td style="padding: 8px 0; color: #64748b; font-weight: 600; width: 40%;">LLM Inference</td>
                        <td style="padding: 8px 0; color: #e2e8f0;">NVIDIA NIM (DeepSeek-V3.1)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(51,65,85,0.3);">
                        <td style="padding: 8px 0; color: #64748b; font-weight: 600;">Embeddings</td>
                        <td style="padding: 8px 0; color: #e2e8f0;">NVIDIA Llama-3.2-NemoRetriever (2048d)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(51,65,85,0.3);">
                        <td style="padding: 8px 0; color: #64748b; font-weight: 600;">Reranking</td>
                        <td style="padding: 8px 0; color: #e2e8f0;">NVIDIA nv-rerank-qa-mixtral-8x7b</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(51,65,85,0.3);">
                        <td style="padding: 8px 0; color: #64748b; font-weight: 600;">NLP</td>
                        <td style="padding: 8px 0; color: #e2e8f0;">spaCy (en_core_web_sm) + NLTK</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(51,65,85,0.3);">
                        <td style="padding: 8px 0; color: #64748b; font-weight: 600;">Retrieval</td>
                        <td style="padding: 8px 0; color: #e2e8f0;">Hybrid BM25 + Cosine Similarity</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(51,65,85,0.3);">
                        <td style="padding: 8px 0; color: #64748b; font-weight: 600;">Embedding Fallback</td>
                        <td style="padding: 8px 0; color: #e2e8f0;">SentenceTransformers (MiniLM-L6-v2)</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #64748b; font-weight: 600;">UI</td>
                        <td style="padding: 8px 0; color: #e2e8f0;">Streamlit + Plotly</td>
                    </tr>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <div class="about-title">🎯 Key Design Decisions</div>
            <div class="about-text" style="line-height: 1.8;">
                <div style="margin-bottom: 8px;">
                    <strong style="color: #7dd3fc;">Atomic Claim Decomposition</strong> — Instead of comparing entire 
                    backstories, we decompose into individual verifiable claims for precise contradiction detection.
                </div>
                <div style="margin-bottom: 8px;">
                    <strong style="color: #7dd3fc;">Single-Pass Contradiction</strong> — A single confirmed contradiction 
                    is sufficient to flag inconsistency, mirroring real-world fact-checking principles.
                </div>
                <div style="margin-bottom: 8px;">
                    <strong style="color: #7dd3fc;">Absence ≠ Contradiction</strong> — The system distinguishes between 
                    "not mentioned" (neutral) and "contradicted" (negative), avoiding false positives from missing context.
                </div>
                <div>
                    <strong style="color: #7dd3fc;">Fallback Retrieval Loop</strong> — When initial evidence is 
                    inconclusive, the system progressively expands the retrieval window before rendering a final verdict.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0; border-top: 1px solid rgba(51,65,85,0.2); margin-top: 2rem;">
        <div style="font-size: 0.75rem; color: #334155;">
            Built with ❤️ for KDSH 2026 Advanced Track · RAG Pipeline Architecture
        </div>
    </div>
    """, unsafe_allow_html=True)
