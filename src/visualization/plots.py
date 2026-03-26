import plotly.graph_objects as go
import pandas as pd


def plot_predictions(df, future_preds):
    fig = go.Figure()

    # historical
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['temperature'],
        mode='lines',
        name='Historical'
    ))

    # future dates
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=len(future_preds)+1, freq='M')[1:]

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_preds,
        mode='lines+markers',
        name='Forecast'
    ))

    fig.update_layout(
        title="Temperature Forecast",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        template="plotly_white"
    )

    return fig