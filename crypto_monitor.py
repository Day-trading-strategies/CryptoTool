import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import ccxt
from config import *

# Page configuration
st.set_page_config(
    page_title="Crypto Price Monitor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

class CryptoPriceMonitor:
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
    
    def create_ohlc_chart(self, df, symbol, timeframe):
        """Create OHLC candlestick chart"""
        if df is None or df.empty:
            return None
        
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=(f'{symbol} Price Chart ({timeframe})',)
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color=BULLISH_COLOR,
                decreasing_line_color=BEARISH_COLOR,
                increasing_fillcolor=BULLISH_COLOR,
                decreasing_fillcolor=BEARISH_COLOR
            ),
            row=1, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - {timeframe} Chart',
            yaxis_title='Price (USDT)',
            xaxis_rangeslider_visible=False,
            height=CHART_HEIGHT,
            showlegend=False,
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(color='white'),
            dragmode='pan'  # Set pan as default drag mode in layout
        )
        
        # Update axes
        fig.update_xaxes(gridcolor='#2d3748', showgrid=True)
        fig.update_yaxes(gridcolor='#2d3748', showgrid=True)
        
        return fig
    
    def display_price_summary(self, selected_cryptos, timeframe):
        """Display price summary cards"""
        cols = st.columns(len(selected_cryptos))
        
        for idx, crypto in enumerate(selected_cryptos):
            symbol = self.available_cryptos[crypto]
            current_price, price_change = self.get_current_price(symbol)
            
            with cols[idx]:
                if current_price is not None:
                    change_color = "🟢" if price_change >= 0 else "🔴"
                    st.metric(
                        label=f"{crypto}/USDT",
                        value=f"${current_price:,.4f}",
                        delta=f"{price_change:+.2f}%"
                    )
                else:
                    st.error(f"Unable to fetch {crypto} data")

def main():
    st.title("🚀 Crypto Price Monitor")
    st.markdown("---")
    
    # Initialize the monitor
    monitor = CryptoPriceMonitor()
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Cryptocurrency selection
    st.sidebar.subheader("Select Cryptocurrencies")
    default_cryptos = DEFAULT_CRYPTOS
    selected_cryptos = st.sidebar.multiselect(
        "Choose cryptocurrencies to monitor:",
        options=list(monitor.available_cryptos.keys()),
        default=default_cryptos
    )
    
    # Timeframe selection
    st.sidebar.subheader("Timeframe")
    selected_timeframe = st.sidebar.selectbox(
        "Select timeframe:",
        options=list(monitor.timeframes.keys()),
        index=4  # Default to 1h
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    
    if not selected_cryptos:
        st.warning("Please select at least one cryptocurrency to monitor.")
        return
    
    # Display price summary
    st.subheader("📊 Price Summary")
    monitor.display_price_summary(selected_cryptos, monitor.timeframes[selected_timeframe])
    
    st.markdown("---")
    
    # Display charts
    st.subheader("📈 OHLC Charts")
    
    # Create tabs for each cryptocurrency
    if len(selected_cryptos) > 1:
        tabs = st.tabs(selected_cryptos)
        for idx, crypto in enumerate(selected_cryptos):
            with tabs[idx]:
                symbol = monitor.available_cryptos[crypto]
                df = monitor.fetch_ohlc_data(symbol, monitor.timeframes[selected_timeframe])
                
                if df is not None:
                    fig = monitor.create_ohlc_chart(df, crypto, selected_timeframe)
                    if fig:
                        st.plotly_chart(
                            fig, 
                            use_container_width=True,
                            config={
                                'displayModeBar': True,
                                'scrollZoom': True,  # Enable mouse scroll zoom
                                'doubleClick': 'reset',  # Double-click to reset zoom
                                'showTips': False,
                                'displaylogo': False,
                                'dragmode': 'pan',  # Set pan as default mode
                                'modeBarButtonsToRemove': [
                                    'downloadPlot',
                                    'toImage',
                                    'lasso2d',
                                    'select2d',
                                    'zoom2d',         # Remove zoom tool
                                    'zoomIn2d',       # Remove zoom in button
                                    'zoomOut2d',      # Remove zoom out button
                                    'autoScale2d'     # Remove auto scale button
                                ]
                            }
                        )
                        
                        # Display data table
                        with st.expander(f"📋 {crypto} Data Table"):
                            st.dataframe(df.tail(10).iloc[::-1], use_container_width=True)
                else:
                    st.error(f"Unable to load chart for {crypto}")
    else:
        # Single cryptocurrency view
        crypto = selected_cryptos[0]
        symbol = monitor.available_cryptos[crypto]
        df = monitor.fetch_ohlc_data(symbol, monitor.timeframes[selected_timeframe])
        
        if df is not None:
            fig = monitor.create_ohlc_chart(df, crypto, selected_timeframe)
            if fig:
                st.plotly_chart(
                    fig, 
                    use_container_width=True,
                    config={
                        'displayModeBar': True,
                        'scrollZoom': True,  # Enable mouse scroll zoom
                        'doubleClick': 'reset',  # Double-click to reset zoom
                        'showTips': False,
                        'displaylogo': False,
                        'dragmode': 'pan',  # Set pan as default mode
                        'modeBarButtonsToRemove': [
                            'downloadPlot',
                            'toImage',
                            'lasso2d',
                            'select2d',
                            'zoom2d',         # Remove zoom tool
                            'zoomIn2d',       # Remove zoom in button
                            'zoomOut2d',      # Remove zoom out button
                            'autoScale2d'     # Remove auto scale button
                        ]
                    }
                )
                
                # Display data table
                with st.expander(f"📋 {crypto} Data Table"):
                    st.dataframe(df.tail(10).iloc[::-1], use_container_width=True)
        else:
            st.error(f"Unable to load chart for {crypto}")
    
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
    
    # Auto-refresh functionality (always enabled)
    time.sleep(AUTO_REFRESH_INTERVAL)
    st.rerun()

if __name__ == "__main__":
    main()
