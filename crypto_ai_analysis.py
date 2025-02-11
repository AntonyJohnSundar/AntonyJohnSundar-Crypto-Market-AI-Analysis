import requests
import pandas as pd
import time
import ccxt
import nltk
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from nltk.sentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
import os
import schedule
import threading
import telegram

# Install dependencies if missing
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

latest_trend = "Loading..."

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = "your_telegram_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=data)

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

# Fetch Market Data with Binance and CoinGecko fallback
def get_market_data(symbol, timeframe='1d'):
    exchange = ccxt.binance()
    for attempt in range(5):  # Retry up to 5 times
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable):
            print(f"Binance API error. Retrying ({attempt+1}/5)...")
            time.sleep(3)
    
    print("Switching to CoinGecko for market data...")
    url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}/market_chart?vs_currency=usd&days=30&interval=daily"
    response = requests.get(url).json()
    if 'prices' in response:
        prices = response['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    raise Exception("Failed to fetch data from both Binance and CoinGecko.")

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

# Predict Next Trend and Schedule Auto-Updates
def update_trend():
    global latest_trend
    try:
        market_data = get_market_data("bitcoin")
        analyzed_data = analyze_trends(market_data)
        model, scaler = get_lstm_model(analyzed_data)
        latest_trend = predict_next_trend(model, scaler, analyzed_data)
        send_telegram_alert(f"📢 Crypto Alert: {latest_trend}")
    except Exception as e:
        latest_trend = f"Error: {e}"

schedule.every(5).minutes.do(update_trend)

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

threading.Thread(target=run_scheduler, daemon=True).start()

# Streamlit UI
st.title("🚀 AI-Powered Crypto Analysis (Live Updates)")
st.write(f"📊 AI Trend Prediction (Live): **{latest_trend}**")
crypto = st.text_input("Enter Crypto (e.g., bitcoin):", "bitcoin")

if st.button("Analyze Now"):
    price = get_crypto_price(crypto)
    sentiment = get_crypto_news_sentiment()
    st.write(f"\n🚀 {crypto.capitalize()} Price: **${price}**")
    st.write(f"📰 News Sentiment: **{'Bullish' if sentiment > 0 else 'Bearish' if sentiment < 0 else 'Neutral'} ({sentiment:.2f})**")
