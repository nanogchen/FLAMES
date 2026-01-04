import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from fileIO import create_zip_download
from flames.q_gen import filter_qpopints_by_range, get_binning_averages
from flames.calc import get_scattering_image

def isf_dsf(u):    

    st.subheader("Intermediate Scattering Function and Dynamic Structure Factor")
    