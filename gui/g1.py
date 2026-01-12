import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math,warnings
from scipy.optimize import curve_fit,OptimizeWarning
import sys
sys.path.insert(0, "../")
from gui.fileIO import create_zip_download
from flames.q_gen import get_binning_averages_by_range,filter_q_points_shell
from flames.calc import get_ISF_corr

# define a single-exp to fit g1
def single_exp(x, tau):
    return np.exp(-x/tau)

# define a double-exp to fit g1
def double_exp(x, tau1, f1, tau2):
    return f1*np.exp(-x/tau1) + (1-f1)*np.exp(-x/tau2)

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
            step=dq,
            key='g1_qrange'
        )
        st.write(f"Start: {q_range[0]} | End: {q_range[1]}")
    with col2:
        ag_str = st.text_input(
            "Select system of interest", 
            # value=f"index 0:{len(u.atoms)//2}", 
            value=f"all", 
            help="MDAnalysis atom group selection",
            key='g1_ag'
            )

    # calc g1
    q_points_shell = filter_q_points_shell(st.session_state.q_values,q_range[0],q_range[1])
    system = u.select_atoms(ag_str)
    system_all = u.select_atoms("all")
    formfact_all = np.array([1.0 if i<system.atoms.n_atoms else 0 for i in range(len(u.atoms))])
    isf = get_ISF_corr(q_points_shell, system_all, u.trajectory[Fr_start:Fr_end:Fr_step], formfact_all)
    
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
        "q": qr_g1,
        "g1":g1_qr[:, :Nt]
    }

    zip_data = create_zip_download(data_to_zip)

    st.download_button(
        label="📥 Download All Results (.zip)",
        data=zip_data,
        file_name=f"g1_results.zip",
        mime="application/zip"
    )

    # fit 
    # fit_func = st.radio("Choose fitting function:", ["single-exp", "double-exp", "triple-exp"], horizontal=True)
    fit_func = st.radio("Choose fitting function:", ["single-exp", "double-exp"], horizontal=True)
    if st.button("Fit"):
        fit_params = []

        # get the time constant for each q
        if fit_func == "single-exp": # tau            

            for iq,ig1 in zip(qr_g1, g1_qr):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", OptimizeWarning)
                    try:
                        
                        popt, pconv = curve_fit(single_exp, time[:Nt], ig1[:Nt], bounds=(0.0, [np.inf]))
                        fit_params.append(popt[0])
                        if np.linalg.cond(pconv) > 1e5: 
                            st.warning(f"Warning: condition number ({np.linalg.cond(pconv)}) is too big!")

                    except OptimizeWarning:
                        pass 

            # plot together: show data as symbols and fit as lines
            fit_params = np.array(fit_params)
            col1,col2,col3 = st.columns([2,1,2])
            with col1:
                fig_col1 = go.Figure()
                idx_q = 0
                for iq, g1 in zip(qr_g1, g1_qr):
                    fig_col1.add_trace(go.Scatter(x=time[:Nt], y=g1[:Nt], mode='markers'))
                    fig_col1.add_trace(go.Scatter(x=time[:Nt], y=single_exp(time[:Nt], fit_params[idx_q]), mode='lines'))
                    idx_q += 1 

                fig_col1.update_layout(
                            autosize=False,
                            title="symbols: data\n\nlines: fit",
                            title_x=0.5,
                            title_xanchor='center',
                            showlegend=False,
                            xaxis_type="log",
                            xaxis_title="dt",
                            yaxis_title="g<sup>(1)</sup>(q,dt)",
                            )
                # # Add a text box outside the plot area (using 'paper' coordinates)
                # fig_col1.add_annotation(
                #             xref="paper", yref="paper", # Use paper coordinates (0,0 bottom-left, 1,1 top-right of the entire figure)
                #             x=0.75, y=0.95,
                #             text="Symbols: data\n\nLines: fit",
                #             showarrow=False,
                #             font=dict(size=20, color="black"),
                #             align="center",
                #             )
                st.plotly_chart(fig_col1, width='content')

            with col3:
                fig_col2 = go.Figure()
                fig_col2.add_trace(go.Scatter(x=qr_g1, y=fit_params, mode='markers'))
                fig_col2.update_layout(
                            autosize=False,
                            title="relaxation time vs. q",
                            title_x=0.5,
                            title_xanchor='center',
                            xaxis_type="log",
                            yaxis_type="log",
                            xaxis_title="q",
                            yaxis_title="τ",
                            )
                st.plotly_chart(fig_col2, width='content')

        elif fit_func == "double-exp": # tau1, f1, tau2
            
            for iq,ig1 in zip(qr_g1, g1_qr):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", OptimizeWarning)
                    try:
                        
                        popt, pconv = curve_fit(double_exp, time[:Nt], ig1[:Nt], bounds=(0.0, [np.inf, 1.0, np.inf]))
                        if popt[0]<popt[2]: # assume tau1>tau2
                            fit_params.append([popt[2],1-popt[1],popt[0]])
                        else:
                            fit_params.append(popt)

                        if np.linalg.cond(pconv) > 1e5: 
                            st.warning(f"Warning: condition number ({np.linalg.cond(pconv)}) is too big!")

                    except OptimizeWarning:
                        st.warning("Optimal parameters not found") 

            # plot together: show data as symbols and fit as lines
            fit_params = np.array(fit_params)
            col1,col2,col3 = st.columns([2,1,2])
            with col1:
                fig_col1 = go.Figure()
                idx_q = 0
                for iq, g1 in zip(qr_g1, g1_qr):
                    fig_col1.add_trace(go.Scatter(x=time[:Nt], y=g1[:Nt], mode='markers'))
                    fig_col1.add_trace(go.Scatter(x=time[:Nt], y=double_exp(time[:Nt], *fit_params[idx_q]), mode='lines'))
                    idx_q += 1 

                fig_col1.update_layout(
                            autosize=False,
                            title="symbols: data\n\nlines: fit",
                            title_x=0.5,
                            title_xanchor='center',
                            showlegend=False,
                            xaxis_type="log",
                            xaxis_title="dt",
                            yaxis_title="g<sup>(1)</sup>(q,dt)",
                            )
                # # Add a text box outside the plot area (using 'paper' coordinates)
                # fig_col1.add_annotation(
                #             xref="paper", yref="paper", # Use paper coordinates (0,0 bottom-left, 1,1 top-right of the entire figure)
                #             x=0.75, y=0.95,
                #             text="Symbols: data\n\nLines: fit",
                #             showarrow=False,
                #             font=dict(size=20, color="black"),
                #             align="center",
                #             )
                st.plotly_chart(fig_col1, width='content')

            with col3:
                fig_col2 = go.Figure()
                fig_col2.add_trace(go.Scatter(x=qr_g1, y=fit_params[:,0], mode='markers', name="time const 1",marker_color='red'))
                fig_col2.add_trace(go.Scatter(x=qr_g1, y=fit_params[:,2], mode='markers', name="time const 2",marker_color='green'))
                fig_col2.update_layout(
                            autosize=False,
                            title="relaxation time vs. q",
                            title_x=0.5,
                            title_xanchor='center',
                            xaxis_type="log",
                            yaxis_type="log",
                            xaxis_title="q",
                            yaxis_title="τ",
                            )
                st.plotly_chart(fig_col2, width='content')


