import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def train_model(df: pd.DataFrame):
    # Features and target
    X = df[['year', 'month']]
    y = df['temperature']

    # Time-based split (important)
    split_index = int(len(df) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, preds)
    print(f"MAE: {mae:.4f}")

    return model