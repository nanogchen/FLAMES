import streamlit as st
import MDAnalysis as mda
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os,math,sys
import tkinter as tk
from tkinter import filedialog
sys.path.insert(0, "../")

from flames.q_gen import get_q_points_all_quads, get_binning_averages
from flames.calc import get_static_sf

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
        exts = ('.xtc', '.lammpstraj', '.pdb', '.gro', '.dcd', '.trr')
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
    st.success(f"Universe Loaded: {len(u.trajectory)} frames, {len(u.atoms)} atoms")
    
    # Analysis Tabs
    tabinit, tab1d, tabpsf, tab2d, tabg1, tabttc = st.tabs([
        "(q,t) Setup", "SAXS 1D", "PSF", "SAXS 2D", "g1 Correlation", "Two-Time Correlation"
    ])

    with tabinit:
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("Wavevector Generation")

            # q_start = st.number_input("q_start (Å⁻¹)", value=0.00, min_value=0.0, step=0.01, format="%.2f")
            L = max(u.dimensions[:3])
            q_end = st.number_input("q_end (Å⁻¹)", value=1.00, min_value=float(2*np.pi/L), step=0.01, format="%.2f")
            max_q_points = st.number_input("Max number of q-points", value=3000, min_value=1000, step=100)
            
            # save input
            st.session_state.input['q_end'] = q_end
            st.session_state.input['max_q_points'] = max_q_points

            # gen q-points
            if st.button("Generate Wavevectors"):
                
                # get box info
                bx, by, bz = u.dimensions[:3]
                st.session_state.input['Box Array'] = np.array([bx, by, bz])

                q_points = get_q_points_all_quads(np.array([bx, by, bz]), q_end, max_points=max_q_points)
                st.session_state.q_values = q_points
                st.session_state.input['dq_values'] = float(2*np.pi/L)

                st.success(f"{q_points.shape[0]} wavevectors generated.")

        with col2:
            st.subheader("Simulation Time")

            frame_start = st.number_input("frame_start", value=0, min_value=0, step=1)
            frame_end = st.number_input("frame_end",     value=1, min_value=frame_start, max_value=len(u.trajectory),step=1)
            frame_step = st.number_input("frame_step",   value=10, min_value=1, step=1)
            traj_dt = st.number_input("traj_dt (ps)", value=1.00, step=0.001, min_value=0.001, format="%.3f")

            st.session_state.input['frame_start'] = frame_start
            st.session_state.input['frame_end'] = frame_end
            st.session_state.input['frame_step'] = frame_step
            st.session_state.input['traj_dt'] = traj_dt

            # gen q-points
            if st.button("OK"):
                st.session_state.dt_values = traj_dt
                st.success(f"Simulation time set.")

        with col3:
            st.subheader("Select Analysis Tasks")
            tasks = st.multiselect(
                "Choose tasks to perform:",
                ["saxs-1D", "PSF", "saxs-2D", "g1 correlation", "ttc"],
                default=st.session_state.selected_tasks
            )
            
            if st.button("Initialize Analysis Pipeline"):
                st.session_state.selected_tasks = tasks
                st.success("Pipeline Initialized!")

            st.text("PSF: partial structure factor\nttc: two-time correlation")
            
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
            st.subheader("1D Scattering Intensity S(q)")
            Fr_start = st.session_state.input['frame_start']
            Fr_end = st.session_state.input['frame_end']
            Fr_step = st.session_state.input['frame_step']
            
            # calculate
            q_points = st.session_state.q_values
            system = u.select_atoms("all")
            formfact_all = np.array([1.0 for _ in range(system.atoms.n_atoms)])
            ssf = get_static_sf(q_points, system, u.trajectory[Fr_start:Fr_end+1:Fr_step], formfact_all)

            num_q_bins = int(st.session_state.input["q_end"]/st.session_state.input["dq_values"])
            qr, ssf_qr = get_binning_averages(num_q_bins, q_end, ssf, q_points)
            ssf_qr_mean = np.mean(ssf_qr, axis=1)

            fig_saxs = px.line(x=qr[1:], y=ssf_qr_mean[1:], 
                log_x=True, log_y=True, 
                markers=True,
                labels={'x':'q', 'y':'S(q)'})
            # --- Download Button ---
            df = pd.DataFrame({
                "q_vector": qr,
                "Intensity": ssf_qr_mean
            })
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download SAXS Data (CSV)",
                data=csv,
                file_name="saxs_analysis.csv",
                mime="text/csv",
            )
            st.plotly_chart(fig_saxs, width='content')

    with tabpsf:
        if check_initialization() and is_ready("PSF"):
            st.header("Point Spread Function Analysis")
            st.info("Calculating the spatial resolution response...")
            r = np.linspace(0, 10, 100)
            psf = np.exp(-r**2 / 2.0)
            st.plotly_chart(px.line(x=r, y=psf, labels={'x':'r (Å)', 'y':'Intensity'}))

    with tab2d:
        if check_initialization() and is_ready("saxs-2D"):
            st.subheader("2D Scattering Intensity S(q)")
            img = np.random.normal(size=(100, 100))
            st.plotly_chart(px.imshow(img, color_continuous_scale='hot'))

    with tabg1:
        if check_initialization() and is_ready("g1 correlation"):
            st.subheader("Dynamics $g^{(1)}(q, dt)$")
            # Allow user to pick from the ALREADY generated q-values
            q_choice = st.select_slider("Select $q$ for correlation analysis", options=np.round(st.session_state.q_values, 3))
            
            tau = np.logspace(0, 4, 100)
            # Relationship: g1 drops faster for higher q (smaller distances)
            g1 = np.exp(-(q_choice**2) * 0.1 * tau)
            
            fig_g1 = px.line(x=tau, y=g1, log_x=True, title=f"Intermediate Scattering Function at q = {q_choice}")
            st.plotly_chart(fig_g1, width='content')

    with tabttc:
        if check_initialization() and is_ready("ttc"):
            st.header("Two-Time Correlation (TTC)")
            # Representation of aging/dynamics
            matrix = np.exp(-np.abs(np.subtract.outer(np.arange(50), np.arange(50))) / 10)
            fig = go.Figure(data=go.Heatmap(z=matrix, colorscale='Viridis'))
            st.plotly_chart(fig, width='content') # or stretch/content

else:
    st.info("Select your MD files from the sidebar to populate analysis panels.")