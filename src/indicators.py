import pandas as pd

def compute_daily_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Compute daily returns for a stock price DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing stock prices with a 'Date' column.
        price_col (str): Column name of stock prices (default 'Close').

    Returns:
        pd.DataFrame: DataFrame with Date and Daily_Return columns.
    """
    df = df.copy()
    df["Daily_Return"] = df[price_col].pct_change()
    return df
