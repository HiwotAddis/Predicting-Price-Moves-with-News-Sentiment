import pandas as pd
from src.indicators import add_indicators

def test_add_indicators():
    # Sample dataframe with minimal stock close prices
    data = {
        "Date": pd.date_range(start="2025-01-01", periods=30, freq="D"),
        "Close": [i for i in range(100, 130)],  # simple increasing close prices
    }
    df = pd.DataFrame(data)

    # Apply indicators
    df_with_indicators = add_indicators(df)

    # Ensure new columns exist
    assert "SMA_20" in df_with_indicators.columns
    assert "RSI" in df_with_indicators.columns
    assert "MACD" in df_with_indicators.columns
    assert "MACD_signal" in df_with_indicators.columns
    assert "MACD_hist" in df_with_indicators.columns

    # Ensure data types are correct (floats)
    assert df_with_indicators["RSI"].dtype == "float64"
    assert df_with_indicators["SMA_20"].dtype == "float64"

    # Check that the number of rows is preserved
    assert len(df_with_indicators) == 30
