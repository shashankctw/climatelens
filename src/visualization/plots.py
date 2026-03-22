import matplotlib.pyplot as plt
import pandas as pd

def plot_predictions(df, future_preds):
    fig, ax = plt.subplots(figsize=(10,5))

    # yearly smoothing
    df_yearly = df.copy()
    df_yearly['year'] = df_yearly['date'].dt.year
    df_yearly = df_yearly.groupby('year')['temperature'].mean().reset_index()

    ax.plot(df_yearly['year'], df_yearly['temperature'], label="Historical (Yearly Avg)")

    # future dates
    last_year = df_yearly['year'].iloc[-1]
    future_years = [last_year + i for i in range(1, len(future_preds)+1)]

   # convert monthly predictions → yearly average
    future_avg = sum(future_preds) / len(future_preds)

    last_year = df_yearly['year'].iloc[-1]
    future_years = [last_year + 1]

    ax.plot(future_years, [future_avg], 'ro', label="Forecast (Yearly Avg)")

    ax.legend()
    ax.set_title("Temperature Forecast (Smoothed)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature")
    ax.set_xlim(df_yearly['year'].min(), future_years[-1])
    return fig