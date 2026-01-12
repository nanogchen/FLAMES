import streamlit as st
import MDAnalysis as mda
import sys,os
sys.path.insert(0, ".")

from gui.traj_load import load_traj
from gui.init_page import init_page
from gui.saxs1d import saxs1d
from gui.saxs2d import saxs2d
from gui.psf import psf
from gui.ttc import ttc
from gui.isf_dsf import isf_dsf
from gui.g1 import g1

# ---------------------------------------------------------------------------- Page Configuration
# ---------------------------------------------------------------------------- Page Configuration
st.set_page_config(layout="wide", page_title="Digital Twin Platform")

# ---------------------------------------------------------------------------- Initialize Session State for Directory
# ---------------------------------------------------------------------------- Initialize Session State for Directory
if "current_path" not in st.session_state:
    st.session_state.current_path = os.getcwd()
if 'u' not in st.session_state:
    st.session_state.u = None    
if "q_values" not in st.session_state:
    st.session_state.q_values = None  # Start as None to force generation step 
if "dt_values" not in st.session_state:
    st.session_state.dt_values = None  # Start as None to force generation step        
if "selected_tasks" not in st.session_state:
    st.session_state.selected_tasks = []
if 'input' not in st.session_state:
    st.session_state.input = {}

# ---------------------------------------------------------------------------- Status Indicator in Sidebar
# ---------------------------------------------------------------------------- Status Indicator in Sidebar
# st.sidebar.markdown("---")
st.sidebar.subheader("Workflow Status")
if st.session_state.u:
    st.sidebar.success("✅ Trajectory Loaded")
else:
    st.sidebar.error("❌ Trajectory Missing")

if st.session_state.q_values is not None:
    st.sidebar.success(f"✅ Q-Vectors Ready ({len(st.session_state.q_values)})")
else:
    st.sidebar.warning("⚠️ Q-Vectors Not Generated")

if st.session_state.dt_values is not None:
    st.sidebar.success(f"✅ Sim Time Info Set")
else:
    st.sidebar.warning("⚠️  Sim Time Info Not Set")

# st.sidebar.markdown("---")
# st.sidebar.subheader("Analysis Task Summary")
# for task in st.session_state.selected_tasks:
#     st.sidebar.write(f"{task}")

def draw_sidebar_footer():
    with st.sidebar:
        st.markdown("---")
        st.caption("Resources & Support")
        
        col1, col2 = st.columns(2)
        with col1:
            st.link_button("GitHub", "https://github.com/", use_container_width=True)
        with col2:
            st.link_button("Issues", "https://github.com/issues", use_container_width=True)
            
        st.link_button("📖 Read The Paper", "https://doi.org/...", use_container_width=True)

draw_sidebar_footer()

# ---------------------------------------------------------------------------- Main Dashboard
# ---------------------------------------------------------------------------- Main Dashboard
st.title("🔬 A Digital Twin for Molecular Scattering")
    
# Analysis Tabs
tabtraj, tabinit, tab1d, tabpsf, tab2d, tabg1, tabisfdsf, tabttc = st.tabs([
    "📂 Trajectory Setup", "(q,t) Setup", "SAXS 1D", "PSF", "SAXS 2D", "g1 Correlation", "ISF-IXS", "Two-Time Correlation"
])

with tabtraj:
    load_traj()

if st.session_state.u:
    with tabinit:
        init_page(st.session_state.u)
                    
    # --- Analysis Tabs (Locked until Step 1 is complete) ---
    def check_initialization():
        if st.session_state.q_values is None:
            st.error("🚨 Action Required: Please go to the '(q,t) Setup' tab and generate your Q-grid first.")
            return False
        if st.session_state.u is None:
            st.error("🚨 Action Required: Please select valid trajectory files in the sidebar.")
            return False
        return True

    # --- Gated Analysis Tabs ---
    def is_ready(task_name):
        if st.session_state.q_values is None:
            st.error("Please initialize Wavevectors in Tab 1 first.")
            return False
        if task_name not in st.session_state.selected_tasks:
            st.warning(f"Task '{task_name}' not selected in (q,t) Setup tab.")
            return False
        return True        

# ---------------------------------------------------------------------------- Tasks
# ---------------------------------------------------------------------------- Tasks

    # ---------------------------------------------------------------------------- saxs-1D
    # ---------------------------------------------------------------------------- saxs-1D
    with tab1d:
        if check_initialization() and is_ready("SAXS-1D"):
            saxs1d(st.session_state.u)            

    # ---------------------------------------------------------------------------- PSF
    # ---------------------------------------------------------------------------- PSF
    with tabpsf:
        if check_initialization() and is_ready("PSF"):
            psf(st.session_state.u)

    # ---------------------------------------------------------------------------- saxs-2D
    # ---------------------------------------------------------------------------- saxs-2D
    with tab2d:
        if check_initialization() and is_ready("SAXS-2D"):
            saxs2d(st.session_state.u)

    # ---------------------------------------------------------------------------- g1
    # ---------------------------------------------------------------------------- g1
    with tabg1:
        if check_initialization() and is_ready("g1 correlation"):
            g1(st.session_state.u)

    # ---------------------------------------------------------------------------- isf-dsf
    # ---------------------------------------------------------------------------- isf-dsf
    with tabisfdsf:
        if check_initialization() and is_ready("ISF-IXS"):
            isf_dsf(st.session_state.u)

    # ---------------------------------------------------------------------------- ttc
    # ---------------------------------------------------------------------------- ttc
    with tabttc:
        if check_initialization() and is_ready("TTC"):
            ttc(st.session_state.u)

else:
    st.info("Select your MD files from the Trajectory Setup tab to populate analysis panels.")