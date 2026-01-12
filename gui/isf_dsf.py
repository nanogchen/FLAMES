import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
import sys
sys.path.insert(0, "../")
from gui.fileIO import create_zip_download
from flames.q_gen import get_binning_averages_by_range,filter_q_points_shell
from flames.calc import get_ISF_corr,fft_dft_symm

def isf_dsf(u):    

    st.subheader("Intermediate Scattering Function and Dynamic Structure Factor")
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
            step=dq,
            key='isf_dsf_qrange'
        )
        st.write(f"Start: {q_range[0]} | End: {q_range[1]}")
    with col2:
        ag_str = st.text_input("Select system of interest", 
            # value=f"index 0:{len(u.atoms)//2}", 
            value=f"all", 
            help="MDAnalysis atom group selection",
            key="isf_dsf_ag"
        )

    # calc isf
    if st.button("Run"):
        q_points_shell = filter_q_points_shell(st.session_state.q_values,q_range[0],q_range[1])
        system = u.select_atoms(ag_str)
        system_all = u.select_atoms("all")
        formfact_all = np.array([1.0 if i<system.atoms.n_atoms else 0 for i in range(len(u.atoms))])
        isf = get_ISF_corr(q_points_shell, system_all, u.trajectory[Fr_start:Fr_end:Fr_step], formfact_all)

        num_q_bins = math.ceil((q_range[1]-q_range[0])/dq)
        qr, isf_qr = get_binning_averages_by_range(num_q_bins, q_range[0], q_range[1], isf, q_points_shell)
        Nt = len(time)//2

        sqw_qr = []
        fig_isf = go.Figure()
        fig_dsf = go.Figure()
        for iq, isf in zip(qr, isf_qr):
            fig_isf.add_trace(go.Scatter(x=time[:Nt], y=isf[:Nt], name=f'q={iq:.2f}'))
            w, sqw = fft_dft_symm(time[:Nt], isf[:Nt])
            sqw_qr.append(sqw)
            fig_dsf.add_trace(go.Scatter(x=w, y=sqw, name=f'q={iq:.2f}'))

        fig_isf.update_layout(
                    autosize=False,
                    xaxis_type="log",
                    xaxis_title="dt",
                    yaxis_title="F(q,dt)",
                    )

        fig_dsf.update_layout(
                    autosize=False,
                    # xaxis_type="log",
                    xaxis_title="ω",
                    yaxis_title=r"S(q,ω)",
                    )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_isf, width='content')

        with col2:
            st.plotly_chart(fig_dsf, width='content')

        # --- Download Button ---
        data_to_zip = {
            "q": qr,
            "Fqt":isf_qr[:, :Nt],
            "w":w,
            "Sqw":np.array(sqw_qr)
        }

        zip_data = create_zip_download(data_to_zip)

        st.download_button(
            label="📥 Download All Results (.zip)",
            data=zip_data,
            file_name=f"isf_dsf.zip",
            mime="application/zip"
        )