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
selected_crypto = "bitcoin"  # Default to Bitcoin

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = "your_telegram_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=data)

# Fetch valid trading symbols from Binance
def get_binance_symbols():
    exchange = ccxt.binance()
    try:
        markets = exchange.load_markets()
        return {symbol.replace("/USDT", "").lower(): symbol for symbol in markets if symbol.endswith("/USDT")}
    except Exception as e:
        print(f"Error fetching Binance symbols: {e}")
        return {}

binance_symbols = get_binance_symbols()

# Validate Crypto Symbol
def validate_crypto_symbol(symbol):
    return symbol.lower() in binance_symbols

# Fetch Crypto Price from CoinGecko
def get_crypto_price(coin):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd"
    response = requests.get(url).json()
    return response.get(coin, {}).get('usd', 'Not Found')

# Fetch Top 10 Potential Gem Coins
def find_gem_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_asc&per_page=50&page=1"
    response = requests.get(url).json()
    coins = sorted(response, key=lambda x: (x['price_change_percentage_24h'], x['market_cap']), reverse=True)[:10]
    return [(coin['name'], coin['symbol'], coin['price_change_percentage_24h']) for coin in coins]

# Fetch Market Data with Binance and CoinGecko fallback
def get_market_data(symbol, timeframe='1d'):
    if symbol not in binance_symbols:
        raise ValueError("Invalid crypto symbol. Try using a different cryptocurrency.")
    binance_symbol = binance_symbols[symbol]
    exchange = ccxt.binance()
    for attempt in range(5):  # Retry up to 5 times
        try:
            bars = exchange.fetch_ohlcv(binance_symbol, timeframe=timeframe, limit=200)
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable):
            print(f"Binance API error. Retrying ({attempt+1}/5)...")
            time.sleep(3)
    raise Exception("Failed to fetch data from Binance.")

# Apply Technical Indicators
def analyze_trends(df):
    if df.shape[0] < 50:
        raise ValueError("Insufficient market data for AI trend analysis. Please try a different cryptocurrency.")
    df['RSI'] = RSIIndicator(df['close']).rsi()
    df['SMA'] = SMAIndicator(df['close'], window=20).sma_indicator()
    macd = MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    return df

# Predict Next Trend
def predict_next_trend(model, scaler, df):
    latest_data = df.iloc[-1:][['RSI', 'SMA', 'MACD', 'MACD_signal']].values
    latest_scaled = scaler.transform(latest_data)
    prediction = model.predict(latest_scaled)[0][0]
    return "Bullish 🚀" if prediction > 0.5 else "Bearish ⚠️"

# Load or Train LSTM AI Model
def get_lstm_model(df):
    model_path = "lstm_model.h5"
    scaler_path = "scaler.npy"
    df = df.dropna()
    features = ['RSI', 'SMA', 'MACD', 'MACD_signal']
    X = df[features].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = np.load(scaler_path, allow_pickle=True).item()
    else:
        model = Sequential([
            LSTM(30, return_sequences=True, input_shape=(X_scaled.shape[1], 1)),
            Dropout(0.1),
            LSTM(30, return_sequences=False),
            Dropout(0.1),
            Dense(15),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_scaled, df['close'], epochs=5, batch_size=16, verbose=0)
        model.save(model_path)
        np.save(scaler_path, scaler)
    return model, scaler

# Streamlit UI
st.title("🚀 AI-Powered Crypto Analysis (Live Updates)")
st.subheader("📊 Market Trend Analysis")
crypto = st.text_input("Enter Crypto (e.g., bitcoin):", "bitcoin")

if st.button("Analyze Now"):
    selected_crypto = crypto.lower()
    if selected_crypto not in binance_symbols:
        st.error("❌ Invalid cryptocurrency symbol. Try a different one.")
    else:
        price = get_crypto_price(selected_crypto)
        market_data = get_market_data(selected_crypto)
        try:
            analyzed_data = analyze_trends(market_data)
            model, scaler = get_lstm_model(analyzed_data)
            trend_prediction = predict_next_trend(model, scaler, analyzed_data)
            st.write(f"\n🚀 {crypto.capitalize()} Price: **${price}**")
            st.write(f"📊 AI Trend Prediction: **{trend_prediction}**")
            st.subheader("💎 Top 10 Potential Gem Coins")
            gem_coins = find_gem_coins()
            for coin in gem_coins:
                st.write(f"- {coin[0]} ({coin[1]}): {coin[2]:.2f}% 24h Change")
        except ValueError as e:
            st.error(str(e))
