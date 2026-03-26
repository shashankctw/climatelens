import sys
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(BASE_DIR, 'src')
sys.path.append(SRC_PATH)

import streamlit as st
import pandas as pd

from src.data.loader import DataLoader
from src.features.build_features import preprocess_data, add_time_features, add_lag_features
from src.models.train import train_model
from src.models.predict import forecast_future
from src.visualization.plots import plot_predictions

st.set_page_config(page_title="ClimateLens", page_icon="🌍", layout="wide")

st.title("🌍 ClimateLens")
st.caption("Explore climate trends and forecast future temperature patterns")

st.info("This tool analyzes historical global temperature data and predicts future trends using machine learning.")

# Sidebar
st.sidebar.header("Controls")
model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Linear Regression"])
steps = st.sidebar.slider("Forecast Months", 3, 36, 12)

# Load
loader = DataLoader()
df = loader.load_csv("climate.csv")
df = loader.standardize_columns(df)

df = preprocess_data(df)
df = add_time_features(df)
df = add_lag_features(df)

model, mae = train_model(df, model_choice)
future = forecast_future(model, df, steps)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Insights", "⚙️ How it Works"])

# ---------- TAB 1 ----------
with tab1:
    st.subheader("Temperature Forecast")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Temp", f"{df['temperature'].mean():.2f}°C")
    col2.metric("Latest Temp", f"{df['temperature'].iloc[-1]:.2f}°C")
    col3.metric("Model Error (MAE)", f"{mae:.2f}")

    fig = plot_predictions(df, future)
    st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 2 ----------
with tab2:
    st.subheader("Insights")

    # Historical change
    historical_change = df['temperature'].iloc[-1] - df['temperature'].iloc[0]

    # Forecast change
    future_change = future[-1] - df['temperature'].iloc[-1]

    col1, col2 = st.columns(2)

    col1.metric("Historical Change", f"{historical_change:.2f}°C")
    col2.metric("Forecast Change", f"{future_change:.2f}°C")

    st.markdown("### 📊 Monthly Pattern")

    monthly_avg = df.groupby("month")["temperature"].mean()
    st.bar_chart(monthly_avg)

    st.markdown("### 🧠 Interpretation")

    if future_change > 0:
        st.write("• Model predicts a rising temperature trend in upcoming months")
    else:
        st.write("• Model predicts stable or decreasing temperatures")

    if abs(future_change) < 0.5:
        st.write("• Changes are relatively mild")
    else:
        st.write("• Noticeable variation expected in future temperatures")

# ---------- TAB 3 ----------
with tab3:
    st.subheader("How It Works")

    st.write("""
    This system uses machine learning to forecast temperature trends.

    Steps:
    1. Load historical data  
    2. Create time-based and previous-temperature features  
    3. Train model (Random Forest / Linear Regression)  
    4. Predict future values  
    5. Visualize results  
    """)