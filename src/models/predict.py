import pandas as pd


def forecast_future(model, df, steps=12):
    future_preds = []

    last_row = df.iloc[-1:].copy()

    for _ in range(steps):
        X = last_row[['year', 'month', 'lag_1', 'lag_2', 'lag_3']]
        pred = model.predict(X)[0]

        future_preds.append(pred)

        # shift lags
        last_row['lag_3'] = last_row['lag_2']
        last_row['lag_2'] = last_row['lag_1']
        last_row['lag_1'] = pred

        # increment month
        month = last_row['month'].values[0] + 1
        year = last_row['year'].values[0]

        if month > 12:
            month = 1
            year += 1

        last_row['month'] = month
        last_row['year'] = year

    return [float(x) for x in future_preds]