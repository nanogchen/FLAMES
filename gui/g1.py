import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from fileIO import create_zip_download
from flames.q_gen import filter_qpopints_by_range, get_binning_averages
from flames.calc import get_scattering_image

def g1(u):

    st.subheader("Dynamics $g^{(1)}(q, dt)$")
    Fr_start = st.session_state.input['frame_start']
    Fr_end = st.session_state.input['frame_end']
    Fr_step = st.session_state.input['frame_step']
    traj_dt = st.session_state.input['traj_dt']

    # The slider returns a tuple (start, end)
    q_range = st.slider(
        label="Select Analysis Q-Range (Å⁻¹)",
        min_value=0.02,
        max_value=st.session_state.input["q_end"],
        value=(0.1, 1.0), # Providing a tuple creates the range bar
        step=0.01
    )
    st.write(f"Start: {q_range[0]} | End: {q_range[1]}")

    # calc g1
    dq = round(st.session_state.input["dq_values"], 2)
    q_range = np.arange(q_range[0],q_range[1],dq)
    time = np.arange(Fr_start, Fr_end, Fr_step)*traj_dt
    g1_all = np.zeros((len(q_range), len(time)))
    
    for idx, iq in enumerate(q_range):                
        # Relationship: g1 drops faster for higher q (smaller distances)
        g1 = np.exp(-(iq**2) * 0.1 * time)
        g1_all[idx, :] = g1
    
    fig_g1 = go.Figure()
    for iq, g1 in zip(q_range, g1_all):
        fig_g1.add_trace(go.Scatter(x=time, y=g1, name=f'q={iq:.2f}'))            

    fig_g1.update_layout(
                autosize=False,
                xaxis_type="log",
                xaxis_title="dt",
                yaxis_title="g<sup>(1)</sup>(q,dt)",
                )
    st.plotly_chart(fig_g1, width='content')

    # --- Download Button ---
    data_to_zip = {
        "q_range": q_range,
        "g1_all":g1_all
    }

    zip_data = create_zip_download(data_to_zip)

    st.download_button(
        label="📥 Download All Results (.zip)",
        data=zip_data,
        file_name=f"g1_results.zip",
        mime="application/zip"
    )
