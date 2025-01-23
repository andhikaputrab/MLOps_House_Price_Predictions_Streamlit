import streamlit as st
import pandas as pd
import plotly.express as px
# import statsmodels.api as sm
import plotly.graph_objects as go
import numpy as np
import json
from src.utils.config import config
from src.utils.styling import load_css
from src.utils.logger import default_logger as logger
from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor

st.set_page_config(
    page_title=config.get('PAGE_TITLE_ANALYTICS'),
    page_icon=config.get('PAGE_ICON_ANALYTICS'),
    layout=config.get('LAYOUT_ANALYTICS')
)

load_css()

# Custom CSS
st.markdown("""
<style>
    /* Custom style for the active tab */
    .stTabs > .tablist > .react-tabs__tab--selected {
        background-color: #0e1117;
        color: #ffffff;
        font-family: 'Courier New', Courier, monospace;
    }
    /* Custom style for all tabs */
    .stTabs > .tablist > .react-tabs__tab {
        background-color: #e8e8e8;
        color: #4f4f4f;
        font-family: 'Courier New', Courier, monospace;
    }
</style>""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def read_dataset():
    logger.info("Initialize data for analytics")
    data_loader = DataLoader()
    
    logger.info("Start load data for analytics")
    df_train, df_test = data_loader.load_data()
    logger.info("Data loaded successfully")
    
    return df_train, df_test

def load_model_artifacts():
    """Load and cache model metrics and feature importance"""
    try:
        metrics = None
        feature_importance = None
        
        # Load metrics if available
        if config.get('METRICS_PATH'):
            with open(config.get('METRICS_PATH'), 'r') as f:
                metrics = json.load(f)
        
        # Load feature importance if available
        if config.get('FEATURE_IMPORTANCE_PATH'):
            with open(config.get('FEATURE_IMPORTANCE_PATH'), 'r') as f:
                feature_importance = json.load(f)
                
        return metrics, feature_importance
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None

try:
    df_train, df_test = read_dataset()
    # X, y = preprocessor.fit_transform(df_train)
    metrics, feature_importance = load_model_artifacts()
    
    if df_train is not None:
        st.title("📊 Data Analytics & Model Performance")

        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "Data Analysis", 
            "Feature Relationships", 
            "Model Performance"
        ])
    
    with tab1:
        st.header("Data Distribution Analysis")
        feature = st.selectbox(
            "Select Feature",
            config.get("FEATURE_COLUMNS")
        )
        
        col1, col2 = st.columns(2)
    
        with col1:
            fig = px.histogram(
                df_train, 
                x=feature,
                marginal="box",
                title=f"Distribution of {feature}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            stats = df_train[feature].describe()
            st.dataframe(stats)
            
    with tab2:
        st.header("Feature Relationships")

        numerical_columns = df_train.select_dtypes(include=['int64', 'float64']).columns
        all_columns = config.get("NUMERICAL_FEATURE_COLUMNS") + config.get("TARGET_COLUMN")
        corr = df_train[all_columns].corr()

        fig = px.imshow(
            corr,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("Select X-axis feature", config.get("FEATURE_COLUMNS"))
                
        with col2:
            y_feature = st.selectbox(
                "Select Y-axis feature", 
                config.get("TARGET_COLUMN"), 
                index=1 if len([config.get("TARGET_COLUMN")]) > 1 else 0
            )
            
        fig = px.scatter(
            df_train,
            x=x_feature,
            y=y_feature,
            title=f"{x_feature} vs {y_feature}",
            # trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("Model Performance")
            
            if metrics and feature_importance:
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Train R² Score", f"{metrics['r2']:.4f}")
                with col2:
                    st.metric("Train RMSE", f"${metrics['rmse']:,.2f}")
                with col3:
                    st.metric("Train MAE", f"${metrics['mae']:,.2f}")
                
                # Training vs Testing Performance
                st.subheader("Training vs Testing Performance")
                metrics_df = pd.DataFrame({
                    'Metric': ['R²', 'RMSE', 'MAE'],
                    'Training': [
                        metrics['rmse'],
                        metrics['mae'],
                        metrics['r2']
                    ],
                    # 'Testing': [
                    #     metrics['test_r2'],
                    #     metrics['test_rmse'],
                    #     metrics['test_mae']
                    # ]
                })
                st.dataframe(metrics_df)

                # Feature Importance Section
                st.subheader("Feature Importance Analysis")
                
                # Create DataFrame from feature importance
                importance_df = pd.DataFrame({
                    'Feature': list(feature_importance.keys()),
                    'Importance': list(feature_importance.values())
                }).sort_values('Importance', ascending=True)

                # Horizontal bar chart
                fig = go.Figure(go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker=dict(
                        color='rgb(26, 118, 255)',
                        line=dict(color='rgba(26, 118, 255, 1.0)', width=1)
                    )
                ))

                fig.update_layout(
                    title='Feature Importance',
                    xaxis_title='Importance Score',
                    yaxis_title='Features',
                    template='plotly_white',
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Feature Importance Details
                with st.expander("Feature Importance Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Top Features")
                        top_features = importance_df.tail(3).copy()
                        top_features['Importance (%)'] = top_features['Importance'] * 100
                        st.dataframe(
                            top_features[['Feature', 'Importance (%)']].round(2)
                        )
                    
                    with col2:
                        st.markdown("### Feature Importance Distribution")
                        fig = px.pie(
                            importance_df,
                            values='Importance',
                            names='Feature',
                            title='Feature Importance Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("Model metrics and feature importance not available. Please train the model first.")
            
except Exception as e:
    logger.info(f"Error in analytics: {str(e)}")
    st.error(f"Error in analytics: {str(e)}")
    