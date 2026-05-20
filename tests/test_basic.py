import pandas as pd
import numpy as np

def test_dataset_load():
    df = pd.read_csv("data/climate.csv")
    assert not df.empty

def test_columns_exist():
    df = pd.read_csv("data/climate.csv")
    required = ["LandAverageTemperature"]

    for col in required:
        assert col in df.columns
        
def forecast(data):
    return np.mean(data)

def test_forecast_output():
    sample = [20, 21, 22, 23]
    result = forecast(sample)
    assert isinstance(result, (int, float))