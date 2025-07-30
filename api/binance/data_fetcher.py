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
    def fetch_ohlc_data_range(_self, symbol, timeframe, start_date, end_date):
        """
        Fetch all OHLC bars from start_date up to end_date,
        without explicitly setting a `limit`—we rely on CCXT’s default.
        """
        all_bars = []
        # turns dates into timestamps
        since_ms = int(start_date.timestamp() * 1_000)
        end_ms   = int(end_date.timestamp()   * 1_000)
        step     = TF_TO_MS[timeframe]

        while since_ms <= end_ms:
            # no `limit` argument here → CCXT/binance uses its default (500)
            ohlcv = _self.exchange.fetch_ohlcv(
                symbol,
                timeframe,
                since=since_ms
            )
            if not ohlcv:
                break

            # keep only bars up to end_date
            page = [bar for bar in ohlcv if bar[0] <= end_ms]
            if not page:
                break

            all_bars.extend(page)
            last_ts = page[-1][0]

            # stop if we’ve already covered end_ms
            if last_ts >= end_ms:
                break

            # advance since to one bar past the last timestamp
            since_ms = last_ts + step

            # if we got fewer than the default page-size (500), we’re done
            if len(ohlcv) < 500:
                break

        # build DataFrame
        df = pd.DataFrame(all_bars, columns=[
            'timestamp','open','high','low','close','volume'
        ])
        df.drop('volume', axis=1, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
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