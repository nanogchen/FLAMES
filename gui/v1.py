import streamlit as st
import MDAnalysis as mda
import tkinter as tk
from tkinter import filedialog
import sys,os
sys.path.insert(0, "../")

from init_page import init_page
from saxs1d import saxs1d
from saxs2d import saxs2d
from psf import psf
from ttc import ttc
from isf_dsf import isf_dsf
from g1 import g1

# ---------------------------------------------------------------------------- Page Configuration
# ---------------------------------------------------------------------------- Page Configuration
st.set_page_config(layout="wide", page_title="MD-XPCS Analyzer")

# ---------------------------------------------------------------------------- Initialize Session State for Directory
# ---------------------------------------------------------------------------- Initialize Session State for Directory
if "current_path" not in st.session_state:
    st.session_state.current_path = os.getcwd()
if "q_values" not in st.session_state:
    st.session_state.q_values = None  # Start as None to force generation step 
if "dt_values" not in st.session_state:
    st.session_state.dt_values = None  # Start as None to force generation step        
if "selected_tasks" not in st.session_state:
    st.session_state.selected_tasks = []
if 'input' not in st.session_state:
    st.session_state.input = {}

# ---------------------------------------------------------------------------- Sidebar: File Navigation
# ---------------------------------------------------------------------------- Sidebar: File Navigation

st.sidebar.title("📁 File Explorer")

# Function to trigger the Tkinter folder picker
def browse_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    root.attributes('-topmost', True)  # Bring the dialog to the front
    directory = filedialog.askdirectory(master=root)
    root.destroy()
    if directory:
        st.session_state.current_path = directory

# Button to load local directory
if st.sidebar.button("📂 Browse Directory"):
    browse_folder()

# Manual path override
st.session_state.current_path = st.sidebar.text_input(
    "Active Path:", 
    st.session_state.current_path
)

# ---------------------------------------------------------------------------- Load Data
# ---------------------------------------------------------------------------- Load Data
@st.cache_resource
def load_trajectory(path, topo, traj):
    if topo and traj:
        try:
            u = mda.Universe(os.path.join(path, topo), os.path.join(path, traj))
            return u
        except Exception as e:
            st.error(f"Failed to load: {e}")
    return None

def list_files(path):
    try:
        # Filter for MD specific formats
        exts = ('.xtc', '.lammpstraj', '.pdb', '.gro', '.data', '.dcd', '.trr')
        files = [f for f in os.listdir(path) if f.lower().endswith(exts)]
        return sorted(files)
    except Exception as e:
        st.sidebar.error(f"Error accessing path: {e}")
        return []

files = list_files(st.session_state.current_path)

if files:
    selected_topo = st.sidebar.selectbox("1. Select Coordinate (PDB/GRO/DATA)", files)
    selected_traj = st.sidebar.selectbox("2. Select Trajectory (XTC/DCD/LAMMPSTRAJ)", files)
    u = load_trajectory(st.session_state.current_path, selected_topo, selected_traj)

else:
    st.sidebar.warning("No MD files found in this directory. Please select your directory first!")

# ---------------------------------------------------------------------------- Status Indicator in Sidebar
# ---------------------------------------------------------------------------- Status Indicator in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Workflow Status")
if files:
    st.sidebar.success("✅ Trajectory Loaded")
else:
    st.sidebar.error("❌ Trajectory Missing")

if st.session_state.q_values is not None:
    st.sidebar.success(f"✅ Q-Vectors Ready ({len(st.session_state.q_values)})")
else:
    st.sidebar.warning("⚠️ Q-Vectors Not Generated")

# if st.session_state.dt_values is not None:
#     st.sidebar.success(f"✅ Time Info Set")
# else:
#     st.sidebar.warning("⚠️  Time Info Not Set")  

# ---------------------------------------------------------------------------- Main Dashboard
# ---------------------------------------------------------------------------- Main Dashboard
st.title("🔬 MD-XPCS Analysis Suite")

if files:
    st.success(f"Trajectory Loaded: {len(u.trajectory)} frames, {len(u.atoms)} atoms")
    
    # Analysis Tabs
    tabinit, tab1d, tabpsf, tab2d, tabg1, tabisfdsf, tabttc = st.tabs([
        "(q,t) Setup", "SAXS 1D", "PSF", "SAXS 2D", "g1 Correlation", "ISF-DSF", "Two-Time Correlation"
    ])

    with tabinit:
        init_page(u)
                    
    # --- Analysis Tabs (Locked until Step 1 is complete) ---
    def check_initialization():
        if st.session_state.q_values is None:
            st.error("🚨 Action Required: Please go to the 'Wavevector Setup' tab and generate your Q-grid first.")
            return False
        if u is None:
            st.error("🚨 Action Required: Please select valid trajectory files in the sidebar.")
            return False
        return True

    # --- Gated Analysis Tabs ---
    def is_ready(task_name):
        if st.session_state.q_values is None:
            st.error("Please initialize Wavevectors in Tab 1 first.")
            return False
        if task_name not in st.session_state.selected_tasks:
            st.warning(f"Task '{task_name}' not selected in Setup tab.")
            return False
        return True        

# ---------------------------------------------------------------------------- Tasks
# ---------------------------------------------------------------------------- Tasks

    # ---------------------------------------------------------------------------- saxs-1D
    # ---------------------------------------------------------------------------- saxs-1D
    with tab1d:
        if check_initialization() and is_ready("saxs-1D"):
            saxs1d(u)            

    # ---------------------------------------------------------------------------- PSF
    # ---------------------------------------------------------------------------- PSF
    with tabpsf:
        if check_initialization() and is_ready("PSF"):
            psf(u)

    # ---------------------------------------------------------------------------- saxs-2D
    # ---------------------------------------------------------------------------- saxs-2D
    with tab2d:
        if check_initialization() and is_ready("saxs-2D"):
            saxs2d(u)

    # ---------------------------------------------------------------------------- g1
    # ---------------------------------------------------------------------------- g1
    with tabg1:
        if check_initialization() and is_ready("g1 correlation"):
            g1(u)

    # ---------------------------------------------------------------------------- isf-dsf
    # ---------------------------------------------------------------------------- isf-dsf
    with tabisfdsf:
        if check_initialization() and is_ready("ISF-DSF"):
            isf_dsf(u)

    # ---------------------------------------------------------------------------- ttc
    # ---------------------------------------------------------------------------- ttc
    with tabttc:
        if check_initialization() and is_ready("ttc"):
            ttc(u)

else:
    st.info("Select your MD files from the sidebar to populate analysis panels.")