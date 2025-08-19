import pandas as pd

def compute_correlation(sentiment_df: pd.DataFrame, returns_df: pd.DataFrame):
    """
    Compute Pearson correlation between sentiment scores and stock returns.

    Parameters
    ----------
    sentiment_df : pd.DataFrame
        DataFrame with columns ['date', 'sentiment'] (daily average sentiment).
    returns_df : pd.DataFrame
        DataFrame with columns ['date', 'return'] (daily stock returns).

    Returns
    -------
    float
        Pearson correlation coefficient.
    """
    # Merge on date
    merged = pd.merge(sentiment_df, returns_df, on="date", how="inner")

    # Compute correlation
    correlation = merged["sentiment"].corr(merged["return"])
    return correlation
