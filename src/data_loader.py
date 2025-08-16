import pandas as pd
import os

def load_news(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['publisher'] = df['publisher'].fillna('Unknown')
    df['headline_length'] = df['headline'].apply(lambda x: len(str(x)))
    df['date_only'] = df['date'].dt.date
    df['hour'] = df['date'].dt.hour
    return df

def load_stock(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    return df
