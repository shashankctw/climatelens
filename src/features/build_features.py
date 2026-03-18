import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only required columns
    df = df[['date', 'temperature']]

    # Convert date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'temperature'])

    # Sort
    df = df.sort_values('date')

    # Handle missing values
    df = df.dropna()

    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    return df

def add_lag_features(df, lags=[1, 2, 3]):
    for lag in lags:
        df[f"lag_{lag}"] = df['temperature'].shift(lag)

    df = df.dropna()
    return df