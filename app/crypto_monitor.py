import streamlit as st
from typing import List

# Import all our components
from api.binance.data_fetcher import BinanceDataFetcher
from charts.ohlc_chart import OHLCChartCreator
from charts.price_summary import PriceSummary
from indicators import (
    HalfTrendIndicator, BollingerBandsIndicator, 
    RSIIndicator, WilliamsRIndicator, KDJIndicator
)
from config import *

class CryptoMonitor:
    """Main application class for the Crypto Price Monitor."""
    
    def __init__(self):
        """Initialize the application components."""
        self._setup_page_config()
        self._initialize_components()
    
    def _setup_page_config(self):
        """Configure Streamlit page settings."""
        st.title("🚀 Crypto Price Monitor")
        st.markdown("---")

        st.set_page_config(
            page_title="Crypto Price Monitor",
            page_icon="₿",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def _initialize_components(self):
        """Initialize all application components."""
        self.data_fetcher = BinanceDataFetcher()
        self.price_display = PriceSummary()
        self.chart_creator = OHLCChartCreator()