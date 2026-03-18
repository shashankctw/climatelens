import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions(df, future_preds):
    plt.figure(figsize=(10,5))

    # historical
    plt.plot(df['date'], df['temperature'], label="Historical")

    # future dates
    last_date = df['date'].iloc[-1]
    future_dates = []

    year = last_date.year
    month = last_date.month

    for _ in future_preds:
        month += 1
        if month > 12:
            month = 1
            year += 1

        future_dates.append(pd.Timestamp(year=year, month=month, day=1))

    # forecast
    plt.plot(future_dates, future_preds, label="Forecast", linestyle='dashed')

    plt.legend()
    plt.title("Temperature Forecast")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()