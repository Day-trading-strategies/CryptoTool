# Configuration file for Crypto Price Monitor

# Default cryptocurrencies to display
DEFAULT_CRYPTOS = ['BTC', 'ETH', 'SOL']

# Available cryptocurrencies (easily expandable)
AVAILABLE_CRYPTOS = {
    'BTC': 'BTC/USDT',
    'ETH': 'ETH/USDT', 
    'SOL': 'SOL/USDT'
}


# Available timeframes
TIMEFRAMES = {
    '1m': '1m',
    '3m': '3m', 
    '5m': '5m',
    '15m': '15m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d'
}

# Indicators that do not overlay on top of price chart.
SEPARATE_AX_INDICATORS ={
    "RSI",
    "William % Range",
    "KDJ"
}

# Chart settings
CHART_HEIGHT = 700
CACHE_TTL_OHLC = 120  # seconds
CACHE_TTL_PRICE = 60  # seconds
AUTO_REFRESH_INTERVAL = 10  # seconds

# Chart colors
BULLISH_COLOR = '#00ff88'
BEARISH_COLOR = '#ff4444'

# Exchange settings
EXCHANGE_NAME = 'binance'
RATE_LIMIT = 1200
ENABLE_RATE_LIMIT = True
