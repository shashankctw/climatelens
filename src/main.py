from data.loader import DataLoader
from config import DATA_FILE, REQUIRED_COLUMNS


def main():
    loader = DataLoader()

    df = loader.load_csv(DATA_FILE)
    loader.validate_columns(df, REQUIRED_COLUMNS)
    
    df =loader.standardize_columns(df)

    print("Data loaded successfully")
    print(df.head())


if __name__ == "__main__":
    main()