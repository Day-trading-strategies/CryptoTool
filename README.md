# Crypto Price Monitor

A real-time cryptocurrency price monitoring web application built with Python, Streamlit, and Plotly. Features OHLC (Open, High, Low, Close) candlestick charts similar to professional trading platforms.

## Features

- 📊 **Real-time OHLC Charts**: Interactive candlestick charts
- ⏱️ **Multiple Timeframes**: 1m, 3m, 5m, 15m, 1h, 4h, 1d
- 💰 **Multiple Cryptocurrencies**: BTC, ETH, SOL (easily expandable to more)
- 🔄 **Auto-refresh**: Automatic data updates every 30 seconds
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🎨 **Professional UI**: Dark theme with intuitive controls

## Default Cryptocurrencies

- **BTC** (Bitcoin)
- **ETH** (Ethereum) 
- **SOL** (Solana)

*Additional cryptocurrencies can be easily added by editing the config.py file*

## Installation

### Quick Setup (Windows)
1. **Clone the repository**
2. **Double-click `setup.bat`** - This will automatically:
   - Create virtual environment
   - Install all dependencies
   - Set everything up for you
3. **Run the app** by double-clicking `run_app.bat`

### Manual Setup
1. **Clone or download** this project to your local machine

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run crypto_monitor.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## Usage

### Interface Overview
- **Sidebar**: Select cryptocurrencies, timeframes, and refresh settings
- **Price Summary**: Quick overview with current prices and 24h changes
- **OHLC Charts**: Professional candlestick charts
- **Data Tables**: Raw OHLC data for detailed analysis

### Controls
- **Cryptocurrency Selection**: Choose which cryptos to monitor
- **Timeframe Selection**: Select chart timeframe (1m to 1d)
- **Auto-refresh**: Toggle automatic updates every 30 seconds
- **Manual Refresh**: Force refresh all data immediately

### Chart Features
- Green candles for price increases
- Red candles for price decreases
- Interactive zoom and pan
- Hover tooltips with OHLC values

## Data Source

- **Exchange**: Binance API via CCXT library
- **Update Frequency**: Real-time with 1-minute cache
- **Data Quality**: Professional-grade OHLC data
- **Rate Limiting**: Built-in to respect API limits

## Extending the App

### Adding New Cryptocurrencies

Edit the `available_cryptos` dictionary in `crypto_monitor.py`:

```python
self.available_cryptos = {
    'BTC': 'BTC/USDT',
    'ETH': 'ETH/USDT', 
    'SOL': 'SOL/USDT',
    'NEW_CRYPTO': 'NEW_CRYPTO/USDT',  # Add here
}
```

### Adding New Timeframes

Edit the `timeframes` dictionary:

```python
self.timeframes = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d',
    '1w': '1w',  # Add here
}
```

## Technical Details

### Architecture
- **Frontend**: Streamlit web framework
- **Charts**: Plotly for interactive visualizations
- **Data**: CCXT library for cryptocurrency exchange APIs
- **Caching**: Streamlit caching for performance optimization

### Performance
- Data caching reduces API calls
- Efficient chart rendering
- Responsive design principles
- Error handling and fallbacks

## Requirements

- Python 3.7+
- Internet connection for real-time data
- Modern web browser

## Troubleshooting

### Common Issues

1. **Installation errors**: Ensure Python 3.7+ is installed
2. **API errors**: Check internet connection
3. **Port conflicts**: Streamlit uses port 8501 by default
4. **Performance**: Disable auto-refresh for slower connections

### Error Messages

- **"Error fetching data"**: Usually network or API issues
- **"Unable to load chart"**: Data fetching problem
- **Empty charts**: No data available for selected timeframe

## License

MIT License - Feel free to modify and distribute.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Future Enhancements

- Technical indicators (MA, RSI, MACD)
- Price alerts and notifications
- Portfolio tracking
- Historical data analysis
- More exchange integrations
- Mobile app version
