import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.insert(0, "../")
from gui.fileIO import create_zip_download
from flames.q_gen import get_binning_averages
from flames.calc import get_scattering_image

def saxs2d(u):

    st.subheader("2D Scattering Intensity S(q1, q2)")
    Fr_start = st.session_state.input['frame_start']
    Fr_end = st.session_state.input['frame_end']
    Fr_step = st.session_state.input['frame_step']
    bx, by, bz = u.dimensions[:3]
    L = max(bx, by, bz) 

    col1, col2, col3 = st.columns(3)
    with col1:
        ag_str = st.text_input("Select system of interest", value="all", help="MDAnalysis atom group selection")
    with col2:
        # select plane            
        st.session_state.input['saxs_2d_plane'] = st.radio("Choose scattering plane:", ["xy", "xz", "yz"], horizontal=True)
    with col3:
        if st.session_state.input['length_unit'] == "real":
            q_temp = 2.0
        else:
            q_temp = 10.0

        q_max = st.number_input("q_max (Å⁻¹ or $\\sigma$)", value=q_temp, min_value=float(2*np.pi/L), step=1.0, format="%.2f")

    # do scattering
    # if st.button("Calculate SAXS-2D"): 
    system = u.select_atoms(ag_str)         
    q_points, ssf_1d, qpts1, qpts2, ssf_2d = get_scattering_image(np.array([bx, by, bz]), q_max, system, 
                                                                u.trajectory[Fr_start:Fr_end:Fr_step], 
                                                                plane=st.session_state.input['saxs_2d_plane'])
    
    num_q_bins = int(q_max/round(2*np.pi/L, 2))
    qr, ssf_r = get_binning_averages(num_q_bins, q_max, ssf_1d, q_points)
    ssf_qr_mean = np.mean(ssf_r, axis=1)
    ssf_2d_mean = np.mean(ssf_2d, axis=-1)
    ssf_2d_mean[int(ssf_2d.shape[0]/2), int(ssf_2d.shape[1]/2)] = 0.0
    
    # st.markdown("##### saxs scattering image (left) and saxs 1d profile (right)")
    fig=go.Figure(go.Heatmap(z=ssf_2d_mean.transpose(), connectgaps=True,
                    zsmooth='best',
                    colorscale='jet', colorbar_thickness=25))

    # Update Layout for better visibility
    fig.update_layout(
                # title=f"saxs scattering image",
                # title_x=0.3,
                # xaxis=dict(
                #     range=[qpts1[0], qpts1[-1]],  # Set X-axis range 
                #     autorange=False # Optional: Explicitly disable auto-ranging
                # ),
                # yaxis=dict(
                #     range=[qpts2[0], qpts2[-1]],  # Set Y-axis range 
                #     autorange=False # Optional: Explicitly disable auto-ranging
                # ),
                autosize=False,
                xaxis_title="q1",
                yaxis_title="q2",
                width=500,  # Set a specific width
                height=500, # Set a specific height to help control the overall figure size
                yaxis_scaleanchor="x"
            )

    st.plotly_chart(fig, width='content')

    # --- Download Button ---
    data_to_zip = {
        "qpts1": qpts1,
        "qpts2": qpts2,
        "saxs_2d":ssf_2d_mean.transpose()
    }

    zip_data = create_zip_download(data_to_zip)

    st.download_button(
        label="📥 Download All Results (.zip)",
        data=zip_data,
        file_name=f"saxs2d_{st.session_state.input['saxs_2d_plane']}_results.zip",
        mime="application/zip"
    )

    if st.button("Get SAXS-1D results"):

        fig_saxs1d = px.line(x=qr[1:], y=ssf_qr_mean[1:], 
            # log_x=True, log_y=True, 
            markers=True,
            labels={'x':'q', 'y':'S(q)'})
        # Update Layout for better visibility
        fig_saxs1d.update_layout(
                    # title=f"saxs 1d profile",
                    # title_x=0.3,
                    autosize=False,
                    # height=500, # Set a specific height to help control the overall figure size
                )              
        st.plotly_chart(fig_saxs1d, width='content')
