"""
Simple test script to verify the crypto monitor app works correctly
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Test imports
    print("Testing imports...")
    import streamlit as st
    import plotly.graph_objects as go
    import pandas as pd
    import ccxt
    from config import *
    print("✅ All imports successful!")
    
    # Test exchange connection
    print("\nTesting exchange connection...")
    exchange = ccxt.binance({
        'sandbox': False,
        'rateLimit': RATE_LIMIT,
        'enableRateLimit': ENABLE_RATE_LIMIT,
    })
    
    # Test fetching a single ticker
    ticker = exchange.fetch_ticker('BTC/USDT')
    print(f"✅ Successfully fetched BTC price: ${ticker['last']:,.2f}")
    
    # Test OHLC data
    print("\nTesting OHLC data...")
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=5)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # Remove volume column as it's not needed
    df = df.drop('volume', axis=1)
    print(f"✅ Successfully fetched OHLC data: {len(df)} candles")
    
    print("\n🎉 All tests passed! The crypto monitor app is ready to run.")
    print("\nTo start the app, run:")
    print('streamlit run crypto_monitor.py')
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install missing dependencies with: pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error: {e}")
    print("There may be an issue with the API or network connection.")
