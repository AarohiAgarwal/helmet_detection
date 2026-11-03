"""
Real-Time Motorcycle Helmet Detection Application.
Professional Streamlit Dashboard for intelligent traffic safety surveillance.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BEST_MODEL_PATH
from src.dashboard import (
    render_overview_page,
    render_live_detection_page,
    render_violation_history_page,
    render_analytics_page,
    render_video_detection_page
)

# 1. Page Configuration
st.set_page_config(
    page_title="AI Helmet Detection & Traffic Safety",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Traffic Monitoring Dark Theme CSS
st.markdown("""
<style>
    /* Main Layout Styling */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    
    /* Metrics Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    div[data-testid="metric-container"] {
        background-color: #1a1a24;
        border: 1px solid #2d2d3f;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.25);
    }
    
    /* Detection Indicator Pills */
    .badge-helmet {
        background-color: #00CC96;
        color: black;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-no-helmet {
        background-color: #EF553B;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-bike {
        background-color: #1f77b4;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar Navigation & Telemetry
st.sidebar.image("https://img.icons8.com/fluency/96/motorcycle.png", width=64)
st.sidebar.title("Traffic AI Monitor")
st.sidebar.caption("Intelligent Motorcycle Helmet Compliance System")

navigation_mode = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Dashboard Overview",
        "📹 Live Detection",
        "📋 Violation History",
        "📈 Analytics & Trends",
        "🎞️ Video Detection"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Engine Status")
model_loaded = BEST_MODEL_PATH.exists()
if model_loaded:
    st.sidebar.success("🟢 YOLOv8n (Custom Trained) Active")
else:
    st.sidebar.warning("🟡 Model Pending Training")

st.sidebar.markdown("""
**Target Classes:**
- 🔵 `BikeWithRider` (Two-wheeler)
- 🟢 `Helmet` (Compliant)
- 🔴 `NoHelmet` (Violation)
""")
st.sidebar.caption("EdgeVision Dataset Architecture © 2026")

# 4. Page Routing
if navigation_mode == "📊 Dashboard Overview":
    render_overview_page()
elif navigation_mode == "📹 Live Detection":
    render_live_detection_page()
elif navigation_mode == "📋 Violation History":
    render_violation_history_page()
elif navigation_mode == "📈 Analytics & Trends":
    render_analytics_page()
elif navigation_mode == "🎞️ Video Detection":
    render_video_detection_page()
