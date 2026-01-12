import streamlit as st
import numpy as np
import sys,json
sys.path.insert(0, "../")
from flames.q_gen import get_q_points_all_quads,get_binning_averages
from flames.calc import get_sf_decomposition

def init_page(u):
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.subheader("Wavevector Generation")

        st.session_state.input['length_unit'] = st.radio("Choose length unit:", ["real", "LJ"], horizontal=True)
        # q_start = st.number_input("q_start (Å⁻¹)", value=0.00, min_value=0.0, step=0.01, format="%.2f")
        L = max(u.dimensions[:3])
        q_end = st.number_input("Max wavenumber q (Å⁻¹ or $\\sigma$⁻¹)", value=1.00, min_value=float(2*np.pi/L), step=0.01, format="%.2f")
        max_q_points = st.number_input("Max number of q-points", value=1000, min_value=1000, step=100)
        
        # save input
        st.session_state.input['q_end'] = q_end
        st.session_state.input['max_q_points'] = max_q_points

        # gen q-points
        if st.button("Generate Wavevectors"):
            
            # get box info
            bx, by, bz = u.dimensions[:3]
            # st.session_state.input['Box Array'] = np.array([bx, by, bz])

            q_points = get_q_points_all_quads(np.array([bx, by, bz]), q_end, max_points=max_q_points)
            st.session_state.q_values = q_points
            st.session_state.input['dq_values'] = float(2*np.pi/L)

            st.success(f"{q_points.shape[0]} wavevectors generated.")

    with col2:
        st.subheader("Simulation Time")

        frame_start = st.number_input("Frame start", value=0, min_value=0, step=1)
        frame_end = st.number_input("Frame end",     value=1, min_value=frame_start, max_value=max(len(u.trajectory)-1, 1),step=1)
        frame_step = st.number_input("Frame step",   value=10, min_value=1, step=1)
        traj_dt = st.number_input("Traj dt (ps or τ)", value=1.00, step=0.001, min_value=0.001, format="%.3f")

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
            ["SAXS-1D", "PSF", "SAXS-2D", "g1 correlation", "ISF-IXS", "TTC"],
            default=st.session_state.selected_tasks
        )
        
        if st.button("Initialize Analysis Pipeline"):
            st.session_state.selected_tasks = tasks
            st.success("Pipeline Initialized!")

            # # save input to file
            # # Convert the dict to a JSON-formatted string
            # json_string = json.dumps(st.session_state.input, indent=4)
            # st.download_button(
            #     label="Download input as JSON",
            #     data=json_string,
            #     file_name="input.json",
            #     mime="application/json"
            # )
        
        st.text("SAXS: small angle X-ray scattering\nPSF: partial structure factor\nISF: intermediate scattering function\nIXS: inelastic X-ray scattering\nTTC: two-time correlation")

