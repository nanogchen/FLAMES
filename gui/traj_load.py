import streamlit as st
import MDAnalysis as mda
import os
import numpy as np
# import tkinter as tk
# from tkinter import filedialog
import tempfile

# # Function to trigger the Tkinter folder picker
# def browse_folder():
#     root = tk.Tk()
#     root.withdraw()  # Hide the main tkinter window
#     root.attributes('-topmost', True)  # Bring the dialog to the front
#     directory = filedialog.askdirectory(master=root)
#     root.destroy()
#     if directory:
#         st.session_state.current_path = directory

# @st.cache_resource
# def load_trajectory(path, topo, traj):
#     if topo and traj:
#         try:
#             u = mda.Universe(os.path.join(path, topo), os.path.join(path, traj))
#             return u
#         except Exception as e:
#             st.error(f"Failed to load: {e}")
#     return None

def list_files(path):
    try:
        # Filter for MD specific formats
        exts = ('.xtc', '.lammpstraj', '.pdb', '.gro', '.data', '.dcd', '.trr')
        files = [f for f in os.listdir(path) if f.lower().endswith(exts)]
        return sorted(files)
    except Exception as e:
        st.sidebar.error(f"Error accessing path: {e}")
        return []

EXAMPLE_DIR = "data"
def get_example_list():
    """Returns a list of folder names inside the examples directory."""
    if os.path.exists(EXAMPLE_DIR):
        return [f for f in os.listdir(EXAMPLE_DIR) if os.path.isdir(os.path.join(EXAMPLE_DIR, f))]
    return []

@st.cache_resource
def load_universe_web(topo, traj, atom_style_str):
    if topo and traj:
        # MDAnalysis needs file paths, so we save uploaded bytes to temp files
        with tempfile.NamedTemporaryFile(suffix=topo.name, delete=False) as tmp_topo:
            tmp_topo.write(topo.getvalue())
            topo_path = tmp_topo.name
            
        with tempfile.NamedTemporaryFile(suffix=traj.name, delete=False) as tmp_traj:
            tmp_traj.write(traj.getvalue())
            traj_path = tmp_traj.name
        
        if atom_style_str is not None:
            return mda.Universe(topo_path, traj_path, atom_style = atom_style_str, format='LAMMPSDUMP')
        else:
            return mda.Universe(topo_path, traj_path)
    return None

def load_traj():

    st.subheader("Upload Trajectory Files")
    st.write("Please upload your topology and trajectory files to begin analysis.")
    
    # ------------------------------------tkinter based for file processing
    # # Button to load local directory
    # if st.button("📂 Browse Directory"):
    #     browse_folder()

    # # Manual path override
    # st.session_state.current_path = st.text_input(
    #     "Active path:", 
    #     st.session_state.current_path
    # )

    # # load files
    # files = list_files(st.session_state.current_path) 

    # col1, col2 = st.columns(2)
    # if files:
    #     with col1:
    #         # topo_file = st.file_uploader("1. Topology (PDB, GRO)", type=['pdb', 'gro'])
    #         selected_topo = st.selectbox("1. Select coordinate (PDB/GRO/DATA)", files)
        
    #     with col2:
    #         # traj_file = st.file_uploader("2. Trajectory (XTC, DCD)", type=['xtc', 'dcd'])
    #         selected_traj = st.selectbox("2. Select trajectory (XTC/DCD/LAMMPSTRAJ)", files)

    # else:
    #     st.warning("No MD files found in this directory. Please select your directory first!")

    # ------------------------------------cloud
    
    # Toggle for Example Mode
    # use_example = st.toggle("💡 Use Example Trajectory", value=False)
    mode = st.radio("Select Data Source:", ["Manual Upload", "Pre-installed Examples"], horizontal=True)
    
    if mode == "Pre-installed Examples":
        examples = get_example_list()

        if examples:

            selected_example = st.selectbox("Choose a system to analyze:", examples)
            ex_path = os.path.join(EXAMPLE_DIR, selected_example)

            col1,col2 = st.columns(2)
            with col1:
                topo_file = st.selectbox("Choose coordinate (PDB/GRO/DATA) to analyze:", list_files(ex_path))
            with col2:
                traj_file = st.selectbox("Choose trajectory (XTC/DCD/GRO/DATA/LAMMPSTRAJ) to analyze:", list_files(ex_path))

            if topo_file and traj_file:

                if traj_file.lower().endswith('lammpstraj') or traj_file.lower().endswith('.dump') or traj_file.lower().endswith('.lammps'):
                    atom_style_str = st.text_input("Atom style for LAMMPS dump file", value="id type x y z", help="LAMMPS dump file format")
                    
                else:
                    atom_style_str = None
                
                st.session_state.input['topo_file'] = topo_file
                st.session_state.input['traj_file'] = traj_file
                if st.button("🚀 Load Example"):
                    if atom_style_str is not None:
                        u = mda.Universe(os.path.join(ex_path, topo_file), 
                                         os.path.join(ex_path, traj_file),
                                         atom_style = atom_style_str, format='LAMMPSDUMP')
                    else:
                        u = mda.Universe(os.path.join(ex_path, topo_file), 
                                         os.path.join(ex_path, traj_file))
                    st.session_state.u = u
                    st.success(f"Example files successfully loaded!")
            else:
                st.error("Missing necessary files in this data folder.")
        else:
            st.error("Example files not found in the 'data' directory.")
            
    else:
        # Manual Upload Mode (your existing code)
        col1, col2 = st.columns(2)
        with col1:
            topo_file = st.file_uploader("Upload coordinate (PDB/GRO/DATA)", type=['pdb', 'gro', 'data'])
        with col2:
            traj_file = st.file_uploader("Upload trajectory (XTC/DCD/GRO/DATA/LAMMPSTRAJ)", type=['xtc', 'dcd', 'gro', 'data', 'lammpstraj', '.dump', '.lammps'])
        
        if topo_file and traj_file:
            if traj_file.name.lower().endswith('lammpstraj') or traj_file.name.lower().endswith('.dump') or traj_file.name.lower().endswith('.lammps'):
                atom_style_str = st.text_input("Atom style for LAMMPS dump file", value="id type x y z", help="LAMMPS dump file format")
            else:
                atom_style_str = None

            st.session_state.input['topo_file'] = topo_file.name.lower()
            st.session_state.input['traj_file'] = traj_file.name.lower()
            st.session_state.input['atom_style'] = atom_style_str
            if st.button("🚀 Load System"):
                # (Your existing tempfile logic here...)
                st.session_state.u = load_universe_web(topo_file, traj_file, atom_style_str)

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

        if hasattr(u.atoms, 'types'):
            unique_names = np.unique(u.atoms.types)

            st.write("Atom Types")
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
# Select atoms by property range
u.select_atoms(\"prop index < 5\")\n
# Select atoms by type
u.select_atoms(\"type 1\")\n
# Select atoms by residue name
u.select_atoms(\"resname DDP\")''')


                    