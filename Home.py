import streamlit as st
from src.utils.config import config
from src.utils.styling import load_css
from src.utils.logger import default_logger as logger
from src.data.data_loader import DataLoader

st.set_page_config(
    page_title=config.get('PAGE_TITLE'),
    page_icon=config.get('PAGE_ICON'),
    layout=config.get('LAYOUT')
)

@st.cache_data
def read_dataset():
    logger.info("Initialize data")
    data_loader = DataLoader()
    
    logger.info("Start load data")
    df_train, df_test = data_loader.load_data()
    logger.info("Data loaded successfully")
    
    return df_train, df_test

load_css()

st.title("🏠 House Price Prediction")
st.markdown("""
**Welcome to the House Price Prediction App!**  

Unlock insights into the housing market with our powerful and interactive platform. Here's what you can do:  
- 🏘️ **Explore** the advanced House Prices dataset from Kaggle.  
- 📊 **Analyze** the relationship between key features and house prices.  
- 🤖 **Predict** house prices using cutting-edge machine learning models.  
- 📈 **Evaluate** model performance with intuitive metrics and visualizations.  
""")

df_train, df_test = read_dataset()

# Make container
st.markdown("### 📊 OVERVIEW DATASET")
col1, col2 = st.columns(2)

# Count row and columns
jumlah_row = df_train.shape[0]
jumlah_columns = df_train.shape[1]

with col1:
    st.metric("Count Row :", jumlah_row)
with col2:
    st.metric("Count Row :", jumlah_columns)
    
st.dataframe(df_train.head(20))