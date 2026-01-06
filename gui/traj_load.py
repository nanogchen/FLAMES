import streamlit as st
import MDAnalysis as mda
# import tempfile
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Function to trigger the Tkinter folder picker
def browse_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    root.attributes('-topmost', True)  # Bring the dialog to the front
    directory = filedialog.askdirectory(master=root)
    root.destroy()
    if directory:
        st.session_state.current_path = directory

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

def load_traj():

    st.subheader("Upload Trajectory Files")
    st.info("Please upload your topology and trajectory files to begin analysis.")
    
    # Button to load local directory
    if st.button("📂 Browse Directory"):
        browse_folder()

    # Manual path override
    st.session_state.current_path = st.text_input(
        "Active path:", 
        st.session_state.current_path
    )

    # load files
    files = list_files(st.session_state.current_path) 

    col1, col2 = st.columns(2)
    if files:
        with col1:
            # topo_file = st.file_uploader("1. Topology (PDB, GRO)", type=['pdb', 'gro'])
            selected_topo = st.selectbox("1. Select coordinate (PDB/GRO/DATA)", files)
        
        with col2:
            # traj_file = st.file_uploader("2. Trajectory (XTC, DCD)", type=['xtc', 'dcd'])
            selected_traj = st.selectbox("2. Select trajectory (XTC/DCD/LAMMPSTRAJ)", files)

    else:
        st.warning("No MD files found in this directory. Please select your directory first!")

    if st.button("🚀 Load System"):
        with st.spinner("Reading MDUniverse..."):
            # # Save to temp files as MDAnalysis requires file paths
            # with tempfile.NamedTemporaryFile(suffix=topo_file.name, delete=False) as tmp_topo:
            #     tmp_topo.write(topo_file.getvalue())
            #     topo_path = tmp_topo.name
                
            # with tempfile.NamedTemporaryFile(suffix=traj_file.name, delete=False) as tmp_traj:
            #     tmp_traj.write(traj_file.getvalue())
            #     traj_path = tmp_traj.name
            
            # Load and store in session state
            u = load_trajectory(st.session_state.current_path, selected_topo, selected_traj)
            st.session_state.u = u
            st.success("System loaded successfully!")

    # Display system info if loaded
    if st.session_state.u:
        u = st.session_state.u
        st.divider()
        st.subheader("System Summary")

        stats_col1, stats_col2, stats_col3 = st.columns(3)
        stats_col1.metric("Atoms", f"{len(u.atoms):,}")        
        stats_col2.metric("Residues", f"{len(u.residues):,}")        
        stats_col3.metric("Frames", f"{len(u.trajectory)}")

        # # If u.atoms has element attributes
        # if hasattr(u.atoms, 'elements'):
        #     # Use .types or .elements to get the string representations
        #     unique_elements = np.unique(u.atoms.elements)
        #     st.write(f"Elements: {', '.join(unique_elements)}")

        if hasattr(u.atoms, 'names'):
            unique_names = np.unique(u.atoms.names)

            st.write("Atom Names")
            st.info(", ".join(unique_names))

        if hasattr(u.atoms, 'resnames'):
            unique_resnames = np.unique(u.atoms.resnames) 

            st.write("Residue Names")
            st.info(", ".join(unique_resnames))

        # atom selection
        st.write("MDAnalysis atom selection examples")
        st.code('''
# Select atoms by index (inclusive, 0-based)
u.select_atoms(\"index 0:5\")\n
# Select atoms by id (inclusive, 1-based)
u.select_atoms(\"id 1:5\")\n
# Select atoms by index range
u.select_atoms(\"prop index < 5\")''')


                    