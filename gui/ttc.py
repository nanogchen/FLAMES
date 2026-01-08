import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.insert(0, "../")
from gui.fileIO import create_zip_download
from flames.q_gen import get_binning_averages_ttc
from flames.calc import get_ttc

def ttc(u):    

    st.subheader("Two-Time Correlation (TTC)")

    Fr_start = st.session_state.input['frame_start']
    Fr_end = st.session_state.input['frame_end']
    Fr_step = st.session_state.input['frame_step']
    bx, by, bz = u.dimensions[:3]
    L = max(bx, by, bz)
    dq = round(2*np.pi/L, 2)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        ag_str = st.text_input(
            "Select system of interest",
            value="all",
            help="MDAnalysis atom group selection",
            key='ttc_ag')
    with col2:
        # select plane            
        st.session_state.input['ttc_2d_plane'] = st.radio(
            "Choose scattering plane:", 
            ["xy", "xz", "yz"], 
            horizontal=True,
            key='ttc_plane'
            )
    with col3:
        Nbins = st.number_input("No. of angular bins", value=18, min_value=10, max_value=36, step=2)
    with col4:
        angle_deg = st.number_input("scattering angle (-180,180]", value=90.0, min_value=-180.0, max_value=180.0, step=1.0, format="%.1f")
    with col5:
        q_i = st.number_input("wavenumber (Å⁻¹ or $\\sigma$⁻¹)", value=0.95, min_value=float(dq)*2, step=float(dq), format="%.2f")
    
    # get ttc: given a q-point and direction (like saxs2d), i.e., localQbin    
    system = u.select_atoms(ag_str)
    formfact_all = np.array([1.0 for _ in range(system.atoms.n_atoms)])
    q_points_bin, ssf, I_q_t1_t2 = get_ttc(np.array([bx, by, bz]), q_i-0.5*dq, q_i+0.5*dq, Nbins, angle_deg,
                            system, u.trajectory[Fr_start:Fr_end:Fr_step], 
                            formfact_all, st.session_state.input['ttc_2d_plane'])

    # do q-average
    qrc, c2 = get_binning_averages_ttc(q_points_bin, ssf, I_q_t1_t2, form="G")

    # --- Download Button ---
    data_to_zip = {
        "qr": qrc,
        "ttc":c2
    }

    zip_data = create_zip_download(data_to_zip)

    st.download_button(
        label="📥 Download All Results (.zip)",
        data=zip_data,
        file_name=f"ttc_{st.session_state.input['ttc_2d_plane']}_results.zip",
        mime="application/zip"
    )

    fig = go.Figure(data=go.Heatmap(z=c2, connectgaps=True,
                    zsmooth='best',
                    zmin=0,  # Set the minimum value for the color scale
                    zmax=1,  # Set the maximum value for the color scale
                    colorscale='jet', colorbar_thickness=25
                    )
                    )
    # Update Layout for better visibility
    fig.update_layout(
                title=f"Two-Time Correlation Function",
                # title_x=0.5,
                autosize=False,
                xaxis_title="t1",
                yaxis_title="t2",
                width=500,  # Set a specific width
                height=500, # Set a specific height to help control the overall figure size
                yaxis_scaleanchor="x"
            )
    st.plotly_chart(fig, width='content') # or stretch/content
