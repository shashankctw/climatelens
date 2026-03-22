import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(BASE_DIR, 'src')
sys.path.append(SRC_PATH)

import streamlit as st
import pandas as pd

from src.data.loader import DataLoader
from src.features.build_features import preprocess_data, add_time_features, add_lag_features
from src.models.train import train_model
from src.models.predict import forecast_future
from src.visualization.plots import plot_predictions


# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="ClimateLens",
    page_icon="🌍",
    layout="wide"
)

# ------------------ HEADER ------------------
st.title("🌍 ClimateLens")
st.caption("Climate Trend Analysis & Forecasting System")

st.markdown("---")


# ------------------ SIDEBAR ------------------
st.sidebar.header("Controls")

steps = st.sidebar.slider("Forecast Months", 3, 24, 12)

show_data = st.sidebar.checkbox("Show Raw Data", False)


# ------------------ LOAD + PROCESS ------------------
@st.cache_data
def load_and_process():
    loader = DataLoader()
    df = loader.load_csv("climate.csv")
    df = loader.standardize_columns(df)

    df = preprocess_data(df)
    df = add_time_features(df)
    df = add_lag_features(df)

    return df


df = load_and_process()


# ------------------ TRAIN ------------------
@st.cache_resource
def train(df):
    return train_model(df)


model = train(df)

future = forecast_future(model, df, steps=steps)


# ------------------ METRICS ------------------
st.markdown("### 📊 Key Insights")

col1, col2, col3 = st.columns(3)

avg_temp = df['temperature'].mean()
latest_temp = df['temperature'].iloc[-1]
trend = "Increasing" if latest_temp > df['temperature'].iloc[0] else "Stable"

col1.metric("Avg Temperature", f"{avg_temp:.2f}°C")
col2.metric("Latest Temperature", f"{latest_temp:.2f}°C")
col3.metric("Trend", trend)


st.markdown("---")


# ------------------ CHART ------------------
st.markdown("### 📈 Temperature Forecast")

fig = plot_predictions(df, future)
st.pyplot(fig)


st.markdown("---")


# ------------------ FORECAST TABLE ------------------
st.markdown("### 🔮 Future Predictions")

forecast_df = pd.DataFrame({
    "Month": [f"Month {i}" for i in range(1, len(future)+1)],
    "Predicted Temperature (°C)": [round(x, 2) for x in future]
})

st.dataframe(forecast_df, use_container_width=True)


# ------------------ OPTIONAL DATA ------------------
if show_data:
    st.markdown("---")
    st.markdown("### 🧾 Processed Data")
    st.dataframe(df.tail(50), use_container_width=True)