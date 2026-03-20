"""
ui_styles.py – Global CSS injected into the Streamlit app.
"""

import streamlit as st


def inject_css():
    st.markdown(
        """
        <style>
        /* ── Font ─────────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
        }

        /* ── Sidebar ──────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: #0f1117;
            border-right: 1px solid #1e2130;
        }
        [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
        [data-testid="stSidebar"] .stRadio label {
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.15s;
        }
        [data-testid="stSidebar"] .stRadio label:hover { background: #1e2130; }

        /* ── Main area ────────────────────────────────────── */
        .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }

        /* ── Metric cards ─────────────────────────────────── */
        [data-testid="metric-container"] {
            background: #1e2130;
            border: 1px solid #2a2f45;
            border-radius: 10px;
            padding: 1rem;
        }

        /* ── Section headers ──────────────────────────────── */
        .section-header {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.1rem;
            font-weight: 600;
            color: #7dd3fc;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        /* ── Badges ───────────────────────────────────────── */
        .badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 99px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.04em;
        }
        .badge-blue  { background:#1d4ed8; color:#bfdbfe; }
        .badge-green { background:#166534; color:#bbf7d0; }
        .badge-amber { background:#92400e; color:#fef3c7; }
        .badge-red   { background:#991b1b; color:#fecaca; }

        /* ── Dataframe tweaks ─────────────────────────────── */
        .stDataFrame { border-radius: 8px; overflow: hidden; }

        /* ── Button accent ────────────────────────────────── */
        .stButton > button {
            background: #2563eb;
            color: #fff;
            border: none;
            border-radius: 7px;
            font-weight: 600;
            transition: background 0.15s, transform 0.1s;
        }
        .stButton > button:hover {
            background: #1d4ed8;
            transform: translateY(-1px);
        }

        /* ── Expander ─────────────────────────────────────── */
        details summary {
            font-weight: 600;
            color: #7dd3fc;
        }

        /* ── Tab labels ───────────────────────────────────── */
        .stTabs [data-baseweb="tab"] {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.82rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
