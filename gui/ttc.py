import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from fileIO import create_zip_download
from flames.q_gen import filter_qpopints_by_range, get_binning_averages_ttc
from flames.calc import get_ttc

def ttc(u):    

    st.subheader("Two-Time Correlation (TTC)")

    Fr_start = st.session_state.input['frame_start']
    Fr_end = st.session_state.input['frame_end']
    Fr_step = st.session_state.input['frame_step']
    q_end = st.session_state.input['q_end']
    q_points = st.session_state.q_values
    bx, by, bz = u.dimensions[:3]
    L = max(bx, by, bz)
    dq = round(2*np.pi/L, 2)
    num_q_bins = int(q_end/dq)

    col1, col2, col3 = st.columns(3)
    with col1:
        ag_str = st.text_input("Select system of interest", value="all", help="MDAnalysis atom group selection")
    with col2:
        # select plane            
        st.session_state.input['ttc_2d_plane'] = st.radio("Choose scattering plane:", ["xy", "xz", "yz"], horizontal=True)
    with col3:
        q_i = st.number_input("q (Å⁻¹ or $\\sigma$)", value=1.00, min_value=float(dq), step=float(dq), format="%.2f")
    
    # get ttc: given a q-point and direction (like saxs2d), i.e., localQbin
    q_points = filter_qpopints_by_range(q_points, q_i-0.5*dq, q_i+0.5*dq)
    system = u.select_atoms("all")
    formfact_all = np.array([1.0 for _ in range(system.atoms.n_atoms)])
    ssf, I_q_t1_t2 = get_ttc(q_points, system, u.trajectory[Fr_start:Fr_end+1:Fr_step], formfact_all)

    # do q-average
    qrc, c2 = get_binning_averages_ttc(q_points, ssf, I_q_t1_t2, form="G")

    # --- Download Button ---
    data_to_zip = {
        "qr": qrc,
        "ttc":c2
    }

    zip_data = create_zip_download(data_to_zip)

    st.download_button(
        label="📥 Download All Results (.zip)",
        data=zip_data,
        file_name=f"ttc_{st.session_state.input['saxs_2d_plane']}_results.zip",
        mime="application/zip"
    )

    fig = go.Figure(data=go.Heatmap(z=c2, colorscale='Viridis'))
    st.plotly_chart(fig, width='content') # or stretch/content
