import requests
import pandas as pd
import time
import ccxt
import nltk
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nltk.sentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

# Install dependencies if missing
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Fetch Crypto Price from CoinGecko
def get_crypto_price(coin):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd"
    response = requests.get(url).json()
    return response.get(coin, {}).get('usd', 'Not Found')

# Fetch Crypto News Sentiment
def get_crypto_news_sentiment():
    url = "https://cryptonews.com/news/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = [h.text.strip() for h in soup.find_all('h4')][:5]
    sentiment_scores = [sia.polarity_scores(headline)['compound'] for headline in headlines]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return avg_sentiment

# Fetch Crypto Market Data (OHLCV) from Binance
def get_binance_data(symbol, timeframe='1d'):
    exchange = ccxt.binance()
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Apply Technical Indicators
def analyze_trends(df):
    df['RSI'] = RSIIndicator(df['close']).rsi()
    df['SMA'] = SMAIndicator(df['close'], window=20).sma_indicator()
    macd = MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    bb = BollingerBands(df['close'])
    df['Bollinger High'] = bb.bollinger_hband()
    df['Bollinger Low'] = bb.bollinger_lband()
    obv = OnBalanceVolumeIndicator(df['close'], df['volume'])
    df['OBV'] = obv.on_balance_volume()
    return df

# Train AI Model for Trend Prediction
def train_ai_model(df):
    df = df.dropna()
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    features = ['RSI', 'SMA', 'MACD', 'MACD_signal', 'Bollinger High', 'Bollinger Low', 'OBV']
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    accuracy = model.score(X_test_scaled, y_test)
    return model, scaler, accuracy

# Predict Next Trend
def predict_next_trend(model, scaler, df):
    latest_data = df.iloc[-1:][['RSI', 'SMA', 'MACD', 'MACD_signal', 'Bollinger High', 'Bollinger Low', 'OBV']]
    latest_scaled = scaler.transform(latest_data)
    prediction = model.predict(latest_scaled)
    return "Bullish 🚀" if prediction[0] == 1 else "Bearish ⚠️"

# Detect Scam Projects using GoPlus API
def check_honeypot(coin_address):
    url = f"https://api.gopluslabs.io/api/v1/token_security/{coin_address}"  # Replace with actual API
    try:
        response = requests.get(url).json()
        if response['result']:
            if response['result']['is_honeypot'] == '1':
                return "⚠️ Scam Detected (Honeypot)"
            if response['result']['is_blacklisted'] == '1':
                return "🚨 Blacklisted Token"
        return "✅ Safe (No Scam Detected)"
    except:
        return "⚠️ Unable to Verify"

# Streamlit UI
st.title("🚀 AI-Powered Crypto Analysis")
crypto = st.text_input("Enter Crypto (e.g., bitcoin):", "bitcoin")
token_address = st.text_input("Enter Token Address for Scam Check:", "0xFakeScamToken123")

if st.button("Analyze"):
    price = get_crypto_price(crypto)
    sentiment = get_crypto_news_sentiment()
    market_data = get_binance_data(f"{crypto.upper()}/USDT")
    analyzed_data = analyze_trends(market_data)
    model, scaler, accuracy = train_ai_model(analyzed_data)
    trend_prediction = predict_next_trend(model, scaler, analyzed_data)
    scam_status = check_honeypot(token_address)
    
    st.write(f"\n🚀 {crypto.capitalize()} Price: **${price}**")
    st.write(f"📰 News Sentiment: **{'Bullish' if sentiment > 0 else 'Bearish' if sentiment < 0 else 'Neutral'} ({sentiment:.2f})**")
    st.write(f"📊 AI Trend Prediction: **{trend_prediction} (Accuracy: {accuracy:.2f})**")
    st.write(f"🔍 Scam Detection: **{scam_status}**")
    
    # Plot Market Data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(market_data['timestamp'], market_data['close'], label='Close Price')
    ax.plot(market_data['timestamp'], market_data['Bollinger High'], linestyle='dashed', color='red', label='Bollinger High')
    ax.plot(market_data['timestamp'], market_data['Bollinger Low'], linestyle='dashed', color='green', label='Bollinger Low')
    ax.legend()
    ax.set_title(f"{crypto.capitalize()} Market Trends")
    st.pyplot(fig)
