import streamlit as st
from src.utils.config import config
from src.utils.styling import load_css
from src.utils.logger import default_logger as logger
from src.data.data_loader import DataLoader

load_css()

st.title("Welcome to House Price Prediction App")
st.markdown("""
**Welcome to the House Price Prediction App!**  

Unlock insights into the housing market with our powerful and interactive platform. Here's what you can do:  
- 🏘️ **Explore** the advanced House Prices dataset from Kaggle.  
- 📊 **Analyze** the relationship between key features and house prices.  
- 🤖 **Predict** house prices using cutting-edge machine learning models.  
- 📈 **Evaluate** model performance with intuitive metrics and visualizations.  
""")