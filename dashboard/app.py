import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.sentiment import analyze_sentiment
from src.indicators import compute_daily_returns
from src.correlation import compute_correlation

# --- Streamlit setup ---
st.set_page_config(page_title="News Sentiment & Stock Market Correlation Dashboard", layout="wide")
st.title("📊 News Sentiment & Stock Market Correlation Dashboard")

# --- Google Drive News Dataset ---
NEWS_FILE_URL = "https://drive.google.com/uc?id=1itNJ1m0FlX1HwcaMrYDvERU91aHMidZS"

@st.cache_data
def load_news_data(nrows=None):
    # nrows can be used to load only a subset if the file is too big
    return pd.read_csv(NEWS_FILE_URL, nrows=nrows, low_memory=False)

st.sidebar.header("⚙️ Settings")
sample_size = st.sidebar.number_input("Rows to load from News CSV", min_value=10000, max_value=300000, step=10000, value=50000)

st.info(f"📥 Loading **{sample_size} rows** from the news dataset hosted on Google Drive.")
news_df = load_news_data(nrows=sample_size)

# --- Upload Stock Dataset ---
stock_file = st.file_uploader("Upload Stock Dataset (CSV)", type="csv")

if stock_file:
    stock_df = pd.read_csv(stock_file)

    st.subheader("1. Uploaded Data Preview")
    st.write("**News Data (first 5 rows):**")
    st.write(news_df.head())
    st.write("**Stock Data (first 5 rows):**")
    st.write(stock_df.head())

    # --- Sentiment Analysis ---
    if "headline" in news_df.columns:
        news_df["sentiment"] = news_df["headline"].astype(str).apply(analyze_sentiment)
    elif "text" in news_df.columns:
        news_df["sentiment"] = news_df["text"].astype(str).apply(analyze_sentiment)
    else:
        st.error("❌ No 'headline' or 'text' column found in news dataset.")
        st.stop()

    news_df["date"] = pd.to_datetime(news_df["date"]).dt.date
    sentiment_daily = news_df.groupby("date")["sentiment"].mean().reset_index()

    # --- Stock Returns ---
    stock_df["Date"] = pd.to_datetime(stock_df["Date"]).dt.date
    stock_df = compute_daily_returns(stock_df, price_col="Close")
    returns_df = stock_df[["Date", "Daily_Return"]].rename(columns={"Date": "date", "Daily_Return": "return"})

    # --- Merge ---
    merged = pd.merge(sentiment_daily, returns_df, on="date", how="inner")

    # --- Correlation ---
    correlation = compute_correlation(sentiment_daily, returns_df)
    st.subheader("2. Correlation Result")
    st.metric("Pearson Correlation", f"{correlation:.3f}")

    # --- Visualizations ---
    st.subheader("3. Visualizations")

    # Sentiment & stock price trends
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(stock_df["Date"], stock_df["Close"], label="Stock Price", color="blue")
    ax.set_ylabel("Price")
    ax2 = ax.twinx()
    ax2.plot(sentiment_daily["date"], sentiment_daily["sentiment"], label="Sentiment", color="orange")
    ax2.set_ylabel("Sentiment Score")
    ax.set_title("Stock Price vs Sentiment Trend")
    st.pyplot(fig)

    # Scatterplot sentiment vs returns
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(merged["sentiment"], merged["return"], alpha=0.5)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Daily Return")
    ax.set_title("Sentiment vs Daily Return")
    st.pyplot(fig)
