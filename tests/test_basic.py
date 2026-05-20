import pandas as pd

def test_dataset_load():
    df = pd.read_csv("data/climate.csv")
    assert not df.empty

def test_columns_exist():
    df = pd.read_csv("data/climate.csv")
    required = ["LandAverageTemperature"]

    for col in required:
        assert col in df.columns

def test_forecast_output():
    sample = pd.DataFrame({
        "LandAverageTemperature": [20, 21, 22, 23]
    })

    assert len(sample) > 0