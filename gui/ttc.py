import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from fileIO import create_zip_download
from flames.q_gen import filter_qpopints_by_range, get_binning_averages
from flames.calc import get_scattering_image

def ttc(u):    

    st.subheader("Two-Time Correlation (TTC)")
    # Representation of aging/dynamics
    matrix = np.exp(-np.abs(np.subtract.outer(np.arange(50), np.arange(50))) / 10)
    fig = go.Figure(data=go.Heatmap(z=matrix, colorscale='Viridis'))
    st.plotly_chart(fig, width='content') # or stretch/content
