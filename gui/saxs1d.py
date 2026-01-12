import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.insert(0, "../")
from flames.q_gen import get_binning_averages
from flames.calc import get_static_sf

def saxs1d(u):
    st.subheader("1D Scattering Intensity S(q)")
    Fr_start = st.session_state.input['frame_start']
    Fr_end = st.session_state.input['frame_end']
    Fr_step = st.session_state.input['frame_step']
    q_end = st.session_state.input['q_end']

    # calculate
    q_points = st.session_state.q_values
    ag_str = st.text_input("system", value="all", help="MDAnalysis atoms selection")
    system = u.select_atoms(ag_str)
    formfact_all = np.array([1.0 for _ in range(system.atoms.n_atoms)])
    ssf = get_static_sf(q_points, system, u.trajectory[Fr_start:Fr_end:Fr_step], formfact_all)

    num_q_bins = int(st.session_state.input["q_end"]/round(st.session_state.input["dq_values"], 2))
    qr, ssf_qr = get_binning_averages(num_q_bins, q_end, ssf, q_points)
    ssf_qr_mean = np.mean(ssf_qr, axis=1)

    fig_saxs = px.line(x=qr[1:], y=ssf_qr_mean[1:], 
        log_x=True, log_y=True, 
        markers=True,
        labels={'x':'q (Å⁻¹)', 'y':'S(q)'}
        )

    # --- Download Button ---
    df = pd.DataFrame({
        "q_vector": qr,
        "Intensity": ssf_qr_mean
    })
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download SAXS-1D Data (CSV)",
        data=csv,
        file_name="saxs_1d_analysis.csv",
        mime="text/csv",
    )
    st.plotly_chart(fig_saxs, width='content')

    # if st.button("Lorentzian fit"):
    #     fit_params = []

