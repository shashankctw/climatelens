import matplotlib.pyplot as plt
import pandas as pd

def plot_predictions(df, future_preds):
    plt.figure(figsize=(10,5))

    # yearly smoothing
    df_yearly = df.copy()
    df_yearly['year'] = df_yearly['date'].dt.year
    df_yearly = df_yearly.groupby('year')['temperature'].mean().reset_index()

    plt.plot(df_yearly['year'], df_yearly['temperature'], label="Historical (Yearly Avg)")

    # future dates
    last_year = df_yearly['year'].iloc[-1]
    future_years = [last_year + i for i in range(1, len(future_preds)+1)]

    plt.plot(future_years, future_preds, linestyle='dashed', label="Forecast")

    plt.legend()
    plt.title("Temperature Forecast (Smoothed)")
    plt.xlabel("Year")
    plt.ylabel("Temperature")
    plt.tight_layout()
    plt.show()