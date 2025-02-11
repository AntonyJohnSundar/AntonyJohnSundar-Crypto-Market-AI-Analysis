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

# Validate Crypto Symbol
def validate_crypto_symbol(symbol):
    url = "https://api.coingecko.com/api/v3/coins/list"
    try:
        response = requests.get(url).json()
        if isinstance(response, list):  # Ensure response is a list
            valid_symbols = {coin.get('id', '') for coin in response if 'id' in coin}
            return symbol.lower() in valid_symbols
    except Exception as e:
        print(f"Error fetching symbol list: {e}")
    return False  # Return False if API fails

# Fetch Crypto Price from CoinGecko
def get_crypto_price(coin):
    if not validate_crypto_symbol(coin):
        return "Invalid symbol"
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd"
    response = requests.get(url).json()
    return response.get(coin, {}).get('usd', 'Not Found')

# Fetch Market Data with Binance and CoinGecko fallback
def get_market_data(symbol, timeframe='1d'):
    if not validate_crypto_symbol(symbol):
        raise ValueError("Invalid crypto symbol")
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
        df['volume'] = 0  # Set default volume to avoid KeyError
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
    
    if 'volume' not in df.columns:
        df['volume'] = 0  # Set default volume if missing
    obv = OnBalanceVolumeIndicator(df['close'], df['volume'])
    df['OBV'] = obv.on_balance_volume()
    return df

# Load or Train LSTM AI Model for Trend Prediction
def get_lstm_model(df):
    model_path = "lstm_model.h5"
    scaler_path = "scaler.npy"
    df = df.dropna()
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    features = ['RSI', 'SMA', 'MACD', 'MACD_signal', 'Bollinger High', 'Bollinger Low', 'OBV']
    X = df[features].values
    y = df['Target'].values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = np.load(scaler_path, allow_pickle=True).item()
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = Sequential([
            LSTM(30, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.1),
            LSTM(30, return_sequences=False),
            Dropout(0.1),
            Dense(15),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_test, y_test), verbose=0)
        model.save(model_path)
        np.save(scaler_path, scaler)
    
    return model, scaler

# Streamlit UI
st.title("🚀 AI-Powered Crypto Analysis (Live Updates)")
crypto = st.text_input("Enter Crypto (e.g., bitcoin):", "bitcoin")

if st.button("Analyze Now"):
    selected_crypto = crypto.lower()
    price = get_crypto_price(selected_crypto)
    market_data = get_market_data(selected_crypto)
    analyzed_data = analyze_trends(market_data)
    model, scaler = get_lstm_model(analyzed_data)
    trend_prediction = predict_next_trend(model, scaler, analyzed_data)
    st.write(f"\n🚀 {crypto.capitalize()} Price: **${price}**")
    st.write(f"📊 AI Trend Prediction: **{trend_prediction}**")
