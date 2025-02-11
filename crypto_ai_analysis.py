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

# Ensure major cryptos are included
binance_symbols.update({"bitcoin": "BTC/USDT", "btc": "BTC/USDT", "ethereum": "ETH/USDT", "eth": "ETH/USDT"})

# Fetch Crypto Price from CoinGecko
def get_crypto_price(coin):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd"
    response = requests.get(url).json()
    return response.get(coin, {}).get('usd', 'Not Found')

# Fetch Top 10 Potential Gem Coins
def find_gem_coins():
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_asc&per_page=50&page=1"
        response = requests.get(url).json()
        coins = sorted(response, key=lambda x: (x.get('price_change_percentage_24h', 0), x.get('market_cap', 0)), reverse=True)[:10]
        return [(coin['name'], coin['symbol'], coin.get('price_change_percentage_24h', 0)) for coin in coins]
    except Exception as e:
        print(f"Error fetching gem coins: {e}")
        return []

# Fetch Market Data with Binance and fallback to CoinGecko
def get_market_data(symbol, timeframe='1d'):
    if symbol not in binance_symbols:
        raise ValueError("Invalid crypto symbol. Try using a different cryptocurrency.")
    binance_symbol = binance_symbols[symbol]
    exchange = ccxt.binance()
    for attempt in range(5):  # Retry up to 5 times
        try:
            bars = exchange.fetch_ohlcv(binance_symbol, timeframe=timeframe, limit=200)
            if bars:
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
    if df.shape[0] < 10:  # Allow AI to work with at least 10 rows instead of 50
        raise ValueError("Insufficient market data for AI trend analysis. Try again later.")
    df['RSI'] = RSIIndicator(df['close']).rsi()
    df['SMA'] = SMAIndicator(df['close'], window=20).sma_indicator()
    macd = MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    return df

# Streamlit UI
st.title("🚀 AI-Powered Crypto Analysis (Live Updates)")
st.subheader("📊 Market Trend Analysis")
crypto = st.text_input("Enter Crypto (e.g., bitcoin, eth, btc):", "bitcoin")

if st.button("Analyze Now"):
    selected_crypto = crypto.lower()
    if selected_crypto not in binance_symbols:
        st.error("❌ Invalid cryptocurrency symbol. Try a different one.")
    else:
        price = get_crypto_price(selected_crypto)
        try:
            market_data = get_market_data(selected_crypto)
            analyzed_data = analyze_trends(market_data)
            st.write(f"\n🚀 {crypto.capitalize()} Price: **${price}**")
            
            # Display Market Trend Chart
            st.subheader("📈 Market Trend")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(market_data['timestamp'], market_data['close'], label='Close Price', color='blue')
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.set_title(f"{crypto.capitalize()} Market Trend")
            ax.legend()
            st.pyplot(fig)
            
            # Display Top 10 Gem Coins
            st.subheader("💎 Top 10 Potential Gem Coins")
            gem_coins = find_gem_coins()
            if gem_coins:
                for coin in gem_coins:
                    st.write(f"- {coin[0]} ({coin[1]}): {coin[2]:.2f}% 24h Change")
            else:
                st.write("No gem coins found. Try again later.")
        except ValueError as e:
            st.error(str(e))
