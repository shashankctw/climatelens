from data.loader import DataLoader
from config import DATA_FILE, REQUIRED_COLUMNS
from features.build_features import preprocess_data, add_time_features
from models.train import train_model
from features.build_features import add_lag_features


def main():
    loader = DataLoader()

    df = loader.load_csv(DATA_FILE)
    loader.validate_columns(df, REQUIRED_COLUMNS)

    df = loader.standardize_columns(df)

    df = preprocess_data(df)
    df = add_lag_features(df)
    df = add_time_features(df)

    model = train_model(df)

    print("Processed Data:")
    print(df.head())


if __name__ == "__main__":
    main()