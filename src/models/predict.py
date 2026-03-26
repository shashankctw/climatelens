import pandas as pd


def forecast_future(model, df, steps=12):
    predictions = []
    df_copy = df.copy()

    for _ in range(steps):
        last = df_copy.iloc[-1]

        next_input = pd.DataFrame([{
            "prev_temp_1": last["temperature"],
            "prev_temp_2": last["prev_temp_1"],
            "prev_temp_3": last["prev_temp_2"],
            "year": last["year"],
            "month": (last["month"] % 12) + 1
        }])

        pred = model.predict(next_input)[0]
        predictions.append(pred)

        new_row = {
            "temperature": pred,
            "prev_temp_1": next_input["prev_temp_1"].values[0],
            "prev_temp_2": next_input["prev_temp_2"].values[0],
            "prev_temp_3": next_input["prev_temp_3"].values[0],
            "year": last["year"] + (1 if next_input["month"].values[0] == 1 else 0),
            "month": next_input["month"].values[0]
        }

        df_copy = pd.concat([df_copy, pd.DataFrame([new_row])], ignore_index=True)

    return predictions