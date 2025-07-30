import streamlit as st
import time

# Import all our components
from api.binance.data_fetcher import BinanceDataFetcher
from app.sidebar.sidebar import Sidebar
from app.charts.ohlc_chart import OHLCChartCreator
from app.charts.price_summary import PriceSummary
from app.backtest_settings.backtest_settings import BacktestSettings
from app.state_manager import SessionStateManager

from app.config import *

class CryptoMonitor:
    """Main application class for the Crypto Price Monitor."""
    
    def __init__(self):
        """Initialize the application components."""
        self.data_fetcher = None
        self.sidebar = None
        self.price_display = None
        self.chart_creator = None
        self.backtest_settings = None
        self.states = SessionStateManager()
    
    def run(self):
        self._setup_page_config()
        self._initialize_components()
        self._render_footer()
        # Auto-refresh functionality (always enabled)
        time.sleep(AUTO_REFRESH_INTERVAL)
        st.rerun()

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
        self.sidebar = Sidebar()
        if not self.sidebar.selected_cryptos:
            st.warning("Please select at least one cryptocurrency to monitor.")
            return
        print(self.states.crypto)
        self.price_display = PriceSummary(self.sidebar.selected_cryptos, self.sidebar.selected_timeframe, self.data_fetcher)
        print(self.sidebar.selected_cryptos, self.sidebar.selected_timeframe, self.sidebar.selected_indicator, self.sidebar.indicator_params)
        self.chart_creator = OHLCChartCreator(self.sidebar.selected_cryptos,
                                               self.sidebar.selected_timeframe,
                                               self.data_fetcher,
                                               self.sidebar.selected_indicator,
                                               self.sidebar.indicator_params, self.states)
        self.backtest_settings = BacktestSettings(self.sidebar.selected_indicator, self.states)
        
    def _render_footer(self):
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666; font-size: 12px;'>
            💡 Data provided by Binance API • Auto-refresh every 10 seconds • Built with Streamlit & Plotly
            </div>
            """, 
            unsafe_allow_html=True
        )