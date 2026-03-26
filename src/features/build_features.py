import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[['date', 'temperature']]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')
    df = df.dropna()
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df['prev_temp_1'] = df['temperature'].shift(1)
    df['prev_temp_2'] = df['temperature'].shift(2)
    df['prev_temp_3'] = df['temperature'].shift(3)

    df = df.dropna()
    return df