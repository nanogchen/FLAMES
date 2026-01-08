import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.insert(0, "../")
from flames.q_gen import get_binning_averages
from flames.calc import get_sf_decomposition

def psf(u):
    st.subheader("Partial structure factors")
    psf_data = {}

    # select two components
    col1, col2 = st.columns(2)
    with col1:
        group_1_str = st.text_input("Component A Selection", value="resname DDP", help="MDAnalysis selection for group A")
    with col2:
        group_2_str = st.text_input("Component B Selection", value="not resname DDP", help="MDAnalysis selection for group B")

    try:
        # Validate selections
        ag1 = u.select_atoms(group_1_str)
        ag2 = u.select_atoms(group_2_str)
        x1 = ag1.atoms.n_atoms / u.atoms.n_atoms
        Fr_start = st.session_state.input['frame_start']
        Fr_end = st.session_state.input['frame_end']
        Fr_step = st.session_state.input['frame_step']
        q_end = st.session_state.input["q_end"]
        
        st.info(f"Group A: {len(ag1)} atoms (molar fraction={x1:.3f}) | Group B: {len(ag2)} atoms\n")                
        if st.button("Calculate PSF for these groups"):                    

            # calculate
            q_points = st.session_state.q_values
            system = u.select_atoms("all")
            formfact_all = np.array([1.0 for _ in range(system.atoms.n_atoms)])
            sf_AA, sf_AB, sf_BB = get_sf_decomposition(q_points,ag1,ag2,u.trajectory[Fr_start:Fr_end:Fr_step])

            num_q_bins = int(q_end/round(st.session_state.input["dq_values"], 2))
            qr, ssf_AA_qr = get_binning_averages(num_q_bins, q_end, sf_AA, q_points)
            qr, ssf_AB_qr = get_binning_averages(num_q_bins, q_end, sf_AB, q_points)
            qr, ssf_BB_qr = get_binning_averages(num_q_bins, q_end, sf_BB, q_points)

            psf_data['q'] = qr
            psf_data['SAA'] = np.mean(ssf_AA_qr, axis=1)
            psf_data['SBB'] = np.mean(ssf_BB_qr, axis=1)
            psf_data['SAB'] = np.mean(ssf_AB_qr, axis=1)

            # other ones
            sf_nn = sf_AA + 2*sf_AB + sf_BB
            sf_cc = (1-x1)**2 * sf_AA + x1**2 * sf_BB - 2*x1*(1-x1)*sf_AB
            sf_nc = (1-x1)*sf_AA - x1*sf_BB + (1-x1-x1)*sf_AB
            qr, ssf_nn_qr = get_binning_averages(num_q_bins, q_end, sf_nn, q_points)
            qr, ssf_cc_qr = get_binning_averages(num_q_bins, q_end, sf_cc, q_points)
            qr, ssf_nc_qr = get_binning_averages(num_q_bins, q_end, sf_nc, q_points)
            psf_data['Snn'] = np.mean(ssf_nn_qr, axis=1)
            psf_data['Scc'] = np.mean(ssf_cc_qr, axis=1)
            psf_data['Snc'] = np.mean(ssf_nc_qr, axis=1)

            # Download PSF Data 
            df_psf = pd.DataFrame(psf_data)           
            csv_psf = df_psf.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download PSF Data", csv_psf, "psf_results.csv", "text/csv")
            
            # plots
            col1, col2 = st.columns(2)
            with col1:
                # Create a Plotly Figure
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=psf_data['q'][1:], y=psf_data['SAA'][1:], mode='lines', 
                                         name='S<sub>AA</sub>', line=dict(color='firebrick', width=2)))            
                fig.add_trace(go.Scatter(x=psf_data['q'][1:], y=psf_data['SBB'][1:], mode='lines', 
                                         name='S<sub>BB</sub>', line=dict(dash='dash', color='royalblue', width=2)))            
                fig.add_trace(go.Scatter(x=psf_data['q'][1:], y=psf_data['SAB'][1:], mode='lines', 
                                         name='S<sub>AB</sub>', line=dict(dash='dot', color='green', width=2)))

                # Update Layout for better visibility
                fig.update_layout(
                    title=f"Multi-component partial structure factors",
                    xaxis_title="q (Å⁻¹)",
                    yaxis_title="S(q)",
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                    hovermode="x unified"
                )

                st.plotly_chart(fig, width='content')

            with col2:
                # Create a Plotly Figure
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=psf_data['q'][1:], y=psf_data['Snn'][1:], mode='lines', 
                                         name='Snn', line=dict(color='firebrick', width=2)))            
                fig.add_trace(go.Scatter(x=psf_data['q'][1:], y=psf_data['Scc'][1:], mode='lines', 
                                         name='Scc', line=dict(dash='dash', color='royalblue', width=2)))            
                fig.add_trace(go.Scatter(x=psf_data['q'][1:], y=psf_data['Snc'][1:], mode='lines', 
                                         name='Snc', line=dict(dash='dot', color='green', width=2)))

                # Update Layout for better visibility
                fig.update_layout(
                    title=f"number/concentration structure factors",
                    xaxis_title="q (Å⁻¹)",
                    yaxis_title="S(q)",
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                    hovermode="x unified"
                )

                st.plotly_chart(fig, width='content')                    
    
    except Exception as e:
        st.error(f"Selection Error: {e}")            
