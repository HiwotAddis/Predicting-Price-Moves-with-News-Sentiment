import pandas as pd
from src.sentiment import compute_sentiment

def test_sentiment_column_added():
    df = pd.DataFrame({"headline": ["Stock rises", "Stock falls"]})
    df = compute_sentiment(df)
    assert "sentiment" in df.columns
    assert df['sentiment'].dtype == float
