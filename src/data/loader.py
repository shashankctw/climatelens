import pandas as pd
from pathlib import Path


class DataLoader:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    def load_csv(self, filename: str) -> pd.DataFrame:
        file_path = self.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)
        return df

    def validate_columns(self, df: pd.DataFrame, required_cols: list):
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
    def standardize_columns(self, df):
        df = df.rename(columns={
        "dt": "date",
        "LandAverageTemperature": "temperature"
    })
        return df