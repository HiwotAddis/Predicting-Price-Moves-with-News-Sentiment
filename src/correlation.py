import pandas as pd
from scipy.stats import pearsonr

def compute_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    df['daily_return'] = df['Close'].pct_change()
    return df

def correlation_analysis(news_df: pd.DataFrame, stock_df: pd.DataFrame):
    daily_sentiment = news_df.groupby('date')['sentiment'].mean().reset_index()
    stock_df = stock_df[['date', 'Close']].sort_values('date')
    stock_df['daily_return'] = stock_df['Close'].pct_change()

    merged = pd.merge(daily_sentiment, stock_df, on='date', how='inner').dropna()
    corr, p_value = pearsonr(merged['sentiment'], merged['daily_return'])
    return corr, p_value, merged
