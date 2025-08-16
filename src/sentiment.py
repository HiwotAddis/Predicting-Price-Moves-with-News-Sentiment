from textblob import TextBlob
import pandas as pd

def compute_sentiment(df: pd.DataFrame, text_column: str = "headline") -> pd.DataFrame:
    df['sentiment'] = df[text_column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return df
