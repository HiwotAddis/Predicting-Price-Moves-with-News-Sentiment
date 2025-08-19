from textblob import TextBlob

def analyze_sentiment(text: str) -> float:
    """
    Analyze sentiment of a given text using TextBlob.
    Returns a polarity score between -1 (negative) and +1 (positive).
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity
