# dashboard.py
# Interactive Streamlit dashboard for supply chain stakeholders

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Supply Chain Forecasting Dashboard",
    page_icon="📦",
    layout="wide"
)

# Database connection
@st.cache_resource
def get_engine():
    """Cached database connection (doesn't reconnect on every interaction)."""
    return create_engine("postgresql://admin:secret@db:5432/supply_chain")

engine = get_engine()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_cv_results():
    """Load model comparison results."""
    try:
        return pd.read_sql("SELECT * FROM cv_results", engine)
    except:
        return pd.DataFrame()

def load_comparison():
    """Load summary comparison."""
    try:
        return pd.read_sql("SELECT * FROM model_comparison", engine)
    except:
        return pd.DataFrame()

def load_forecasts():
    """Load future forecasts."""
    try:
        return pd.read_sql(
            "SELECT * FROM forecasts ORDER BY forecast_date DESC LIMIT 100",
            engine
        )
    except:
        return pd.DataFrame()

def load_inventory():
    """Load inventory recommendations."""
    try:
        return pd.read_sql(
            "SELECT * FROM inventory_recommendations ORDER BY generated_at DESC LIMIT 1",
            engine
        )
    except:
        return pd.DataFrame()

def load_sales_history():
    """Load recent sales for plotting."""
    query = """
    SELECT transaction_date, units_sold
    FROM sales_transactions
    WHERE product_id = 1 AND store_id = 1
    ORDER BY transaction_date DESC
    LIMIT 365
    """
    try:
        df = pd.read_sql(query, engine)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        return df.sort_values('transaction_date')
    except:
        return pd.DataFrame()

# ============================================================
# DASHBOARD LAYOUT
# ============================================================

st.title("📦 Supply Chain Forecasting Dashboard")
st.markdown("Multi-model demand forecasting with inventory optimization")

# Sidebar
st.sidebar.header("Controls")
selected_model = st.sidebar.selectbox(
    "Select Model to View",
    ["All Models", "ARIMA", "Prophet", "XGBoost"]
)

# ============================================================
# TOP METRICS ROW
# ============================================================

st.header("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

cv_df = load_cv_results()
inv_df = load_inventory()

with col1:
    if not cv_df.empty:
        best_mape = cv_df.groupby('model')['mape'].mean().min()
        st.metric("Best MAPE", f"{best_mape:.2f}%")
    else:
        st.metric("Best MAPE", "N/A")

with col2:
    if not inv_df.empty:
        safety = inv_df.iloc[0]['safety_stock_units']
        st.metric("Safety Stock", f"{safety:.0f} units")
    else:
        st.metric("Safety Stock", "N/A")

with col3:
    if not inv_df.empty:
        reorder = inv_df.iloc[0]['reorder_point_units']
        st.metric("Reorder Point", f"{reorder:.0f} units")
    else:
        st.metric("Reorder Point", "N/A")

with col4:
    if not inv_df.empty:
        service = inv_df.iloc[0]['service_level'] * 100
        st.metric("Service Level", f"{service:.0f}%")
    else:
        st.metric("Service Level", "N/A")

# ============================================================
# MODEL COMPARISON CHART
# ============================================================

st.header("Model Comparison")

if not cv_df.empty:
    # MAPE comparison box plot
    fig_comparison = px.box(
        cv_df,
        x='model',
        y='mape',
        color='model',
        title="MAPE Distribution Across CV Windows (Lower is Better)",
        labels={'mape': 'MAPE (%)', 'model': 'Model'}
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Metrics table
    st.subheader("Detailed Metrics")
    comparison = cv_df.groupby('model').agg({
        'mae': 'mean',
        'rmse': 'mean',
        'mape': ['mean', 'std']
    }).round(2)
    st.dataframe(comparison, use_container_width=True)
else:
    st.warning("No model comparison data available. Run the pipeline first.")

# ============================================================
# SALES HISTORY + FORECAST
# ============================================================

st.header("Sales History & Forecast")

sales_df = load_sales_history()
forecast_df = load_forecasts()

if not sales_df.empty:
    fig = go.Figure()
    
    # Historical sales
    fig.add_trace(go.Scatter(
        x=sales_df['transaction_date'],
        y=sales_df['units_sold'],
        mode='lines',
        name='Actual Sales',
        line=dict(color='blue', width=1)
    ))
    
    # Forecasts
    if not forecast_df.empty:
        forecast_df['forecast_date'] = pd.to_datetime(forecast_df['forecast_date'])
        fig.add_trace(go.Scatter(
            x=forecast_df['forecast_date'],
            y=forecast_df['predicted_units'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title="Actual Sales vs Forecast",
        xaxis_title="Date",
        yaxis_title="Units Sold",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No sales data available.")

# ============================================================
# INVENTORY STATUS
# ============================================================

st.header("Inventory Status")

if not inv_df.empty:
    inv = inv_df.iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Safety Stock**
        {inv['safety_stock_units']:.0f} units
        
        Buffer for forecast uncertainty
        """)
    
    with col2:
        st.success(f"""
        **Reorder Point**
        {inv['reorder_point_units']:.0f} units
        
        Order when inventory hits this level
        """)
    
    with col3:
        st.warning(f"""
        **Max Inventory**
        {inv['max_inventory_units']:.0f} units
        
        Ceiling to prevent overstocking
        """)
    
    # Inventory gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=inv['service_level'] * 100,
        title={'text': "Service Level Target"},
        gauge={
            'axis': {'range': [80, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [80, 90], 'color': "lightgray"},
                {'range': [90, 95], 'color': "yellow"},
                {'range': [95, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 95
            }
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
else:
    st.warning("No inventory recommendations available. Run the pipeline first.")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("Built with Python, PostgreSQL, XGBoost, and Streamlit | Supply Chain Optimization Project")