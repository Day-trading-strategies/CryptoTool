import ccxt
import streamlit as st
import pandas as pd

from app.config import *

class BinanceDataFetcher:
    def __init__(self):
        self.exchange = ccxt.binance({
            'sandbox': False,
            'rateLimit': RATE_LIMIT,
            'enableRateLimit': ENABLE_RATE_LIMIT,
        })
        
        # Available cryptocurrencies (from config)
        self.available_cryptos = AVAILABLE_CRYPTOS
        
        # Timeframe mapping (from config)
        self.timeframes = TIMEFRAMES
    
        self.separate_ax_indicators = SEPARATE_AX_INDICATORS

    @st.cache_data(ttl=CACHE_TTL_OHLC, show_spinner=False)  # Hide spinner
    def fetch_ohlc_data(_self, symbol, timeframe, limit=100):
        """Fetch OHLC data from exchange"""
        try:
            ohlcv = _self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:  # Check if data is empty
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # Remove volume column as it's not needed
            df = df.drop('volume', axis=1)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            # Silently return None to prevent UI disruption
            return None
    
    @st.cache_data(ttl=CACHE_TTL_PRICE, show_spinner=False)  # Hide spinner
    def get_current_price(_self, symbol):
        """Get current price for a symbol"""
        try:
            ticker = _self.exchange.fetch_ticker(symbol)
            if not ticker or 'last' not in ticker:  # Check if ticker data is valid
                return None, None
            return ticker['last'], ticker.get('percentage', 0)
        except Exception as e:
            # Silently return None to prevent UI disruption
            return None, None