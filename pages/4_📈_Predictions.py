import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from src.utils.config import config
from src.utils.styling import load_css

st.set_page_config(
    page_title=config.get('PAGE_TITLE_PREDICTION'), 
    page_icon=config.get('PAGE_ICON_PREDICTION'), 
    layout=config.get('LAYOUT_PREDICTION'))

load_css()

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
    
st.title("🏠 House Price Prediction")

st.markdown("""
-Coming Soon-
""")