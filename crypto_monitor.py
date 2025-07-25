import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import ta
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

    # returns changed dataframe with new data 'half_trend' that will be used to creating half trend indicator
    def calculate_half_trend(self, df, period=10, multiplier=1):
        # Half Trend calculation based on ATR
        hl2 = (df['high'] + df['low']) / 2
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()
        # Initialize with first True Range
        ht = [hl2.iloc[0]]
        trend = True  # True=uptrend, False=downtrend
        for i in range(1, len(df)):
            delta = atr.iloc[i] * multiplier
            prev_ht = ht[-1]
            if hl2.iloc[i] > prev_ht + delta:
                ht.append(prev_ht + delta)
                trend = True
            elif hl2.iloc[i] < prev_ht - delta:
                ht.append(prev_ht - delta)
                trend = False
            else:
                ht.append(prev_ht)
        df['half_trend'] = ht

        return df

    def create_ohlc_chart(self, df, symbol, timeframe, indicators, params):
        """Create OHLC candlestick chart"""
        if df is None or df.empty:
            return None
        
        # Compute Half Trend if selected
        if 'Half Trend' in indicators:
            df = self.calculate_half_trend(df, period=params['Half Trend']['period'],
                                           multiplier=params['Half Trend']['multiplier'])


        sep_inds = []
        for ind in indicators:
            if ind in self.separate_ax_indicators:
                sep_inds.append(ind)

        # Changes height of chart as more indicators are added.
        n_rows = 1 + len(sep_inds)
        weights      = [3] + [1] * len(sep_inds)
        total_weight = sum(weights)
        row_heights  = [w/total_weight for w in weights]

        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes = True,
            vertical_spacing=0.02,
            row_heights = row_heights,
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
        fig.update_yaxes(title_text='Price (USDT)', row=1, col=1)

        # Plot Half Trend
        if 'Half Trend' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'], 
                    y=df['half_trend'],
                    mode='lines', 
                    name='Half Trend',
                    line=dict(width=1)
                ), row=1, col=1
            )

        # Bollinger Band
        if "Bollinger Band" in indicators:
            w = params['Bollinger Band']['window']
            dev = params['Bollinger Band']['window_dev']
            bb = ta.volatility.BollingerBands(close=df["close"], window=w, window_dev=dev)

            df["bb_middle"] = bb.bollinger_mavg()
            df["bb_upper"]  = bb.bollinger_hband()
            df["bb_lower"]  = bb.bollinger_lband()

            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["bb_middle"],
                    mode="lines",
                    line=dict(color="rgba(255,0,0,1)", width=1),
                    opacity = 0.6,
                    name="BB Middle (20)"
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["bb_upper"],
                    mode="lines",
                    line=dict(color="rgba(51,153,255,0.8)", width=1),
                    name="BB Upper (20,+2σ)",
                    hovertemplate="Upper: %{y:.2f}<extra></extra>"
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["bb_lower"],
                    mode="lines",
                    line=dict(color="rgba(51,153,255,0.8)", width=1),
                    name="BB Lower (20,−2σ)",
                    hovertemplate="Lower: %{y:.2f}<extra></extra>"
                ),
                row=1, col=1
            )
        
        for idx, ind in enumerate(sep_inds, start=2):

            # RSI Chart
            if ind == "RSI":
                w = params['RSI']['window']
                df["rsi"] = (ta.momentum.RSIIndicator(df["close"], window=w).rsi())

                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['rsi'],
                        mode='lines',
                        line=dict(width=1),
                        name='RSI (14)',
                        hovertemplate='RSI: %{y:.1f}<extra></extra>'
                    ),
                    row=idx, col=1
                )
                fig.update_yaxes(title_text='RSI', row=idx, col=1)

            # William % Range chart
            if ind == "William % Range":
                lbp = params['William % Range']['lbp']
                df["WR"] = (ta.momentum.williams_r(df["high"], df["low"], df["close"], lbp=lbp))

                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['WR'],
                        mode='lines',
                        line=dict(width=1),
                        name='William % Range (14)',
                        hovertemplate='%{y:.1f}<extra></extra>'
                    ),
                    row=idx, col=1
                )
                fig.update_yaxes(title_text='Williams % Range', row=idx, col=1)
                
            # KDJ chart
            if ind == "KDJ":
                period = params['KDJ']['period']
                signal = params['KDJ']['signal']
                stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=period, smooth_window=signal)
                df['%K'] = stoch.stoch()
                df['%D'] = stoch.stoch_signal()
                df['%J'] = 3 * df['%K'] - 2 * df['%D']

                fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["%K"],
                    mode="lines",
                    line=dict(color="rgba(255,0,0,1)", width=1),
                    opacity = 0.6,
                    name="K (20)"
                ),
                row=idx, col=1
                ) 
                fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["%D"],
                    mode="lines",
                    line=dict(color="rgba(51,153,255,0.8)", width=1),
                    name="D",
                ),
                row=idx, col=1
                )
                fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["%J"],
                    mode="lines",
                    line=dict(color="rgba(51,153,255,0.8)", width=1),
                    name="J",
                ),
                row=idx, col=1
                )
                fig.update_yaxes(title_text='KDJ', row=idx, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - {timeframe} Chart',
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
    #indicator selection
    st.sidebar.subheader("Indicator")
    selected_indicator = st.sidebar.multiselect(
        "Select Indicator:",
        options=["RSI", "Bollinger Band", "KDJ", "Half Trend", "William % Range"]
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

    st.sidebar.markdown("---")

    # Indicator Parameters that appear when indicator is selected
    indicator_params = {}

    if "RSI" in selected_indicator:
        indicator_params["RSI"] = {
            "window": st.sidebar.number_input("RSI window", 2, 100, 14, key="RSI_w")
        }

    if "Bollinger Band" in selected_indicator:
        indicator_params["Bollinger Band"] = {
            "window":     st.sidebar.number_input("BB window", 2, 100, 20, key="BB_w"),
            "window_dev": st.sidebar.slider("BB σ-dev", 1.0, 4.0, 2.0, 0.1, key="BB_d")
        }

    if "KDJ" in selected_indicator:
        indicator_params["KDJ"] = {
            "period": st.sidebar.number_input("KDJ period", 2, 100, 14, key="KDJ_p"),
            "signal": st.sidebar.number_input("KDJ signal", 1, 10, 3, key="KDJ_s"),
        }

    if "William % Range" in selected_indicator:
        indicator_params["William % Range"] = {
            "lbp": st.sidebar.number_input("William %R window", 2, 100, 14, key="WR_w")
        }

    if "Half Trend RMA" in selected_indicator:
        indicator_params["Half Trend"] = {
            "period":     st.sidebar.number_input("HT ATR period", 2, 50, 10, key="HT_p"),
            "multiplier": st.sidebar.slider("HT multiplier", 0.1, 3.0, 1.0, 0.1, key="HT_m")
        }

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
                    fig = monitor.create_ohlc_chart(df, crypto, selected_timeframe, selected_indicator, indicator_params)
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
        
        # if able to load coin price data.
        if df is not None:
            fig = monitor.create_ohlc_chart(df, crypto, selected_timeframe, selected_indicator)
            if fig:
                st.plotly_chart(
                    fig, 
                    use_container_width=True,
                    config={
                        'displayModeBar': True,
                        'modeBarButtonsToAdd': [
                            ['drawrect']
                        ],
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
