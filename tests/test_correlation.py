import pandas as pd
from src.correlation import compute_daily_returns, correlation_analysis

def test_compute_daily_returns():
    # Minimal stock dataframe
    data = {
        "date": pd.date_range(start="2025-01-01", periods=5, freq="D"),
        "Close": [100, 102, 101, 105, 107],
    }
    df = pd.DataFrame(data)

    result = compute_daily_returns(df)

    # Ensure daily_return column exists
    assert "daily_return" in result.columns

    # First value should be NaN
    assert pd.isna(result["daily_return"].iloc[0])

    # Later values should be floats
    assert isinstance(result["daily_return"].iloc[1], float)


def test_correlation_analysis():
    # Mock news dataset
    news_data = {
        "date": pd.date_range(start="2025-01-01", periods=5, freq="D"),
        "sentiment": [0.1, -0.2, 0.3, 0.0, -0.1],
    }
    news_df = pd.DataFrame(news_data)

    # Mock stock dataset
    stock_data = {
        "date": pd.date_range(start="2025-01-01", periods=5, freq="D"),
        "Close": [100, 101, 102, 104, 103],
    }
    stock_df = pd.DataFrame(stock_data)

    corr, p_value, merged = correlation_analysis(news_df, stock_df)

    # Ensure correlation and p_value are floats
    assert isinstance(corr, float)
    assert isinstance(p_value, float)

    # Ensure merged dataframe is not empty
    assert not merged.empty

    # Ensure expected columns are in merged dataset
    assert "sentiment" in merged.columns
    assert "daily_return" in merged.columns
