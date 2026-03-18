from data.loader import DataLoader
from config import DATA_FILE, REQUIRED_COLUMNS
from features.build_features import preprocess_data, add_time_features
from models.train import train_model
from features.build_features import add_lag_features
from models.predict import forecast_future
from visualization.plots import plot_predictions


def main():
    loader = DataLoader()

    df = loader.load_csv(DATA_FILE)
    loader.validate_columns(df, REQUIRED_COLUMNS)

    df = loader.standardize_columns(df)

    df = preprocess_data(df)
    df = add_lag_features(df)
    df = add_time_features(df)

    model = train_model(df)
    future = forecast_future(model, df)
    plot_predictions(df, future)

    print("Processed Data:")
    print(df.head())

    print("Future Predictions:")
    for i, val in enumerate(future, 1):
        print(f"Month {i}: {val:.2f}°C")


if __name__ == "__main__":
    main()