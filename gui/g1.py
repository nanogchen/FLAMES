import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
from fileIO import create_zip_download
from flames.q_gen import get_binning_averages_by_range,filter_q_points_shell
from flames.calc import get_ISF_corr

def g1(u):

    st.subheader("Dynamics $g^{(1)}(q, dt)$")
    Fr_start = st.session_state.input['frame_start']
    Fr_end = st.session_state.input['frame_end']
    Fr_step = st.session_state.input['frame_step']
    traj_dt = st.session_state.input['traj_dt']
    time = np.arange(Fr_start, Fr_end, Fr_step)*traj_dt
    dq = round(st.session_state.input["dq_values"], 2)

    col1, col2 = st.columns(2)
    with col1:
        # The slider returns a tuple (start, end)
        q_range = st.slider(
            label="Select analysis q-range (Å⁻¹)",
            min_value=dq,
            max_value=st.session_state.input["q_end"],
            value=(dq, min(dq*5, st.session_state.input["q_end"])), # Providing a tuple creates the range bar
            step=dq
        )
        st.write(f"Start: {q_range[0]} | End: {q_range[1]}")
    with col2:
        ag_str = st.text_input("Select system of interest", value=f"index 0:{len(u.atoms)//2}", help="MDAnalysis atom group selection")

    # calc g1
    q_points_shell = filter_q_points_shell(st.session_state.q_values,q_range[0],q_range[1])
    system = u.select_atoms(ag_str)
    system_all = u.select_atoms("all")
    formfact_all = np.array([1.0 if i<system.atoms.n_atoms else 0 for i in range(len(u.atoms))])
    isf = get_ISF_corr(q_points_shell, system_all, u.trajectory[Fr_start:Fr_end+1:Fr_step], formfact_all)
    
    g1 = np.zeros(isf.shape)
    for idx in range(isf.shape[0]):
        g1[idx, :] = isf[idx,:]/isf[idx,0]

    num_q_bins = math.ceil((q_range[1]-q_range[0])/dq)
    qr_g1, g1_qr = get_binning_averages_by_range(num_q_bins, q_range[0], q_range[1], g1, q_points_shell)
    Nt = len(time)//2

    fig_g1 = go.Figure()
    for iq, g1 in zip(qr_g1, g1_qr):
        fig_g1.add_trace(go.Scatter(x=time[:Nt], y=g1[:Nt], name=f'q={iq:.2f}'))            

    fig_g1.update_layout(
                autosize=False,
                xaxis_type="log",
                xaxis_title="dt",
                yaxis_title="g<sup>(1)</sup>(q,dt)",
                )
    st.plotly_chart(fig_g1, width='content')

    # --- Download Button ---
    data_to_zip = {
        "q": qr_g1[:Nt],
        "g1":g1_qr[:Nt]
    }

    zip_data = create_zip_download(data_to_zip)

    st.download_button(
        label="📥 Download All Results (.zip)",
        data=zip_data,
        file_name=f"g1_results.zip",
        mime="application/zip"
    )

    # fit 
    fit_func = st.radio("Choose fitting function:", ["single-exp", "double-exp", "triple-exp"], horizontal=True)
    if st.button("Fit"):
        # show data as symbols and fit as lines
        # get the time constant for each q

        pass
        # func fit

