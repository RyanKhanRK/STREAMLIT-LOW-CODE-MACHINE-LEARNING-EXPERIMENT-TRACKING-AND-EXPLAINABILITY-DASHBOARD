"""
Low-Code ML Experiment Tracking & Explainability Dashboard
=========================================================
Main Streamlit application entry point.
Run with: streamlit run app.py
"""

import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="ML Experiment Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Internal modules ─────────────────────────────────────────────────────────
from modules.ui_styles import inject_css
from modules.session import init_session
from pages.data_upload   import render as render_data
from pages.preprocessing import render as render_preprocess
from pages.model_train   import render as render_train
from pages.experiment_tracker import render as render_tracker
from pages.explainability    import render as render_explain

# ── Bootstrap ────────────────────────────────────────────────────────────────
inject_css()
init_session()

# ── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧪 ML Dashboard")
    st.markdown("---")

    PAGES = {
        "📂 Data Upload":          "data",
        "⚙️ Preprocessing":        "preprocess",
        "🤖 Train & Evaluate":     "train",
        "📊 Experiment Tracker":   "tracker",
        "🔍 Explainability (SHAP)":"explain",
    }

    selected = st.radio(
        "Navigate",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )
    page_key = PAGES[selected]

    st.markdown("---")
    # Dataset status pill
    if st.session_state.get("df") is not None:
        df = st.session_state["df"]
        st.success(f"✅ Dataset loaded\n{df.shape[0]} rows × {df.shape[1]} cols")
    else:
        st.info("No dataset loaded yet.")

    if st.session_state.get("experiments"):
        n = len(st.session_state["experiments"])
        st.success(f"📋 {n} experiment run(s) recorded")

    st.markdown("---")
    st.caption("Built with Streamlit · scikit-learn · SHAP · MLflow")

# ── Route to the right page ──────────────────────────────────────────────────
if page_key == "data":
    render_data()
elif page_key == "preprocess":
    render_preprocess()
elif page_key == "train":
    render_train()
elif page_key == "tracker":
    render_tracker()
elif page_key == "explain":
    render_explain()
