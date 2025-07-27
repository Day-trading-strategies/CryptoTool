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
import pytz

# Page configuration
st.set_page_config(
    page_title="Crypto Price Monitor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS for button stability
st.markdown("""
<style>
    /* Keep navigation buttons stable */
    .stButton > button {
        width: 100%;
        height: 38px !important;
        min-height: 38px !important;
        max-height: 38px !important;
    }
    
    /* Prevent column shifting */
    div[data-testid="column"] {
        min-height: 50px !important;
        display: flex;
        align-items: center;
    }
</style>""", unsafe_allow_html=True)

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
            # Convert to local timezone
            local_tz = datetime.now().astimezone().tzinfo
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(local_tz)
            # Remove timezone info to avoid display issues
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
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

    def create_ohlc_chart(self, df, symbol, timeframe, indicators, params, highlighted_timestamps=None, chart_position=None):
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
        
        # Add highlighting overlays using scatter markers based on timestamps
        if highlighted_timestamps:
            price_range = df['high'].max() - df['low'].min()
            for timestamp in highlighted_timestamps:
                # Find the row with matching timestamp
                matching_rows = df[df['timestamp'] == timestamp]
                
                # If exact match not found, find the closest earlier timestamp
                if matching_rows.empty:
                    # Find all timestamps that are earlier than or equal to the target
                    earlier_timestamps = df[df['timestamp'] <= timestamp]['timestamp']
                    if not earlier_timestamps.empty:
                        # Get the latest (closest) timestamp that's still earlier
                        closest_timestamp = earlier_timestamps.max()
                        matching_rows = df[df['timestamp'] == closest_timestamp]
                        is_approximate = True
                    else:
                        # If no earlier timestamp found, skip this highlight
                        continue
                else:
                    is_approximate = False
                
                if not matching_rows.empty:
                    candle = matching_rows.iloc[0]
                    # Add a star marker slightly above the highlighted candlestick
                    star_y = candle['high'] + (price_range * 0.05)  # 5% above high
                    
                    # Different colors for exact vs approximate matches
                    if is_approximate:
                        star_color = '#FFA500'  # Orange for approximate matches
                        border_color = '#FF6347'  # Tomato border
                        hover_suffix = f'<br><i>Approximate match (closest earlier)</i>'
                    else:
                        star_color = '#FFD700'  # Gold for exact matches
                        border_color = '#FF8C00'  # Orange border
                        hover_suffix = ''
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[candle['timestamp']],
                            y=[star_y],
                            mode='markers',
                            marker=dict(
                                symbol='star',
                                size=20,
                                color=star_color,
                                line=dict(width=2, color=border_color)
                            ),
                            name=f'Highlight {timestamp.strftime("%H:%M:%S")}',
                            showlegend=False,
                            hovertext=f'Highlighted Candlestick<br>Target: {timestamp.strftime("%Y-%m-%d %H:%M:%S")}<br>Actual: {candle["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}<br>High: ${candle["high"]:.4f}<br>Close: ${candle["close"]:.4f}{hover_suffix}'
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
        
        # Auto-pan chart based on navigation position (takes priority) or highlighted timestamps
        if chart_position is not None:
            # Use navigation position (chart_position is an index into the dataframe)
            if 0 <= chart_position < len(df):
                end_time = df.iloc[chart_position]['timestamp']
                
                # Calculate time window based on timeframe (show about 60 candles before position)
                timeframe_minutes = {
                    '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                    '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720, '1d': 1440
                }
                
                # Get minutes for current timeframe, default to 60 if not found
                tf_minutes = timeframe_minutes.get(timeframe, 60)
                
                # Show approximately 60 candles worth of data before the position
                time_window_minutes = tf_minutes * 60
                start_time = end_time - pd.Timedelta(minutes=time_window_minutes)
                
                # Ensure start_time is not before our data range
                data_start = df['timestamp'].min()
                if start_time < data_start:
                    start_time = data_start
                
                # Set the x-axis range to focus on the navigation position
                fig.update_layout(
                    xaxis=dict(
                        range=[start_time, end_time],
                        type='date'
                    )
                )
        elif highlighted_timestamps:
            # Only use highlight positioning if no navigation position is set
            # Find the latest highlighted timestamp and resolve it to actual chart data
            latest_highlight = max(highlighted_timestamps)
            
            # Find the actual timestamp to use (exact match or closest earlier)
            matching_rows = df[df['timestamp'] == latest_highlight]
            if matching_rows.empty:
                # Find closest earlier timestamp (this is what actually gets displayed)
                earlier_timestamps = df[df['timestamp'] <= latest_highlight]['timestamp']
                if not earlier_timestamps.empty:
                    end_time = earlier_timestamps.max()  # Use the actual candle timestamp
                else:
                    end_time = latest_highlight  # Fallback to original if no earlier found
            else:
                end_time = latest_highlight  # Exact match found
            
            # Calculate time window based on timeframe (show about 50-80 candles before highlight)
            timeframe_minutes = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720, '1d': 1440
            }
            
            # Get minutes for current timeframe, default to 60 if not found
            tf_minutes = timeframe_minutes.get(timeframe, 60)
            
            # Show approximately 60 candles worth of data before the highlight
            time_window_minutes = tf_minutes * 60
            start_time = end_time - pd.Timedelta(minutes=time_window_minutes)
            
            # Ensure start_time is not before our data range
            data_start = df['timestamp'].min()
            if start_time < data_start:
                start_time = data_start
            
            # Set the x-axis range to focus on the highlighted area
            fig.update_layout(
                xaxis=dict(
                    range=[start_time, end_time],
                    type='date'
                )
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

    if "Half Trend" in selected_indicator:
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
                    # Get highlighted candlesticks for this crypto (using timestamps)
                    highlight_key = f"highlighted_candles_{crypto}"
                    highlighted_timestamps = st.session_state.get(highlight_key, [])
                    
                    # Chart navigation state
                    nav_key = f"chart_nav_{crypto}"
                    timeframe_key = f"timeframe_{crypto}"
                    
                    # Check if timeframe has changed
                    previous_timeframe = st.session_state.get(timeframe_key, None)
                    current_timeframe = selected_timeframe
                    timeframe_changed = previous_timeframe != current_timeframe
                    
                    if timeframe_changed:
                        # Reset navigation to latest when timeframe changes
                        st.session_state[nav_key] = len(df) - 1
                        st.session_state[timeframe_key] = current_timeframe
                    
                    if nav_key not in st.session_state:
                        st.session_state[nav_key] = len(df) - 1  # Start at the most recent candlestick
                    
                    current_position = int(st.session_state[nav_key])
                    
                    # Check if we should auto-pan to a new highlight
                    should_auto_pan_to_highlight = False
                    navigation_was_updated = False
                    if highlighted_timestamps and not timeframe_changed:  # Don't auto-pan if timeframe just changed
                        # Check if this is a new highlight by comparing with previous state
                        prev_highlights_key = f"prev_highlighted_candles_{crypto}"
                        prev_highlights = st.session_state.get(prev_highlights_key, [])
                        
                        # If we have new highlights, auto-pan to the latest one
                        if len(highlighted_timestamps) > len(prev_highlights):
                            # Find the latest highlighted timestamp and set navigation position to it
                            latest_highlight = max(highlighted_timestamps)
                            
                            # Find the closest matching row in the dataframe (exact or closest earlier)
                            matching_rows = df[df['timestamp'] == latest_highlight]
                            if matching_rows.empty:
                                # Find closest earlier timestamp - this is what actually gets displayed
                                earlier_timestamps = df[df['timestamp'] <= latest_highlight]['timestamp']
                                if not earlier_timestamps.empty:
                                    closest_timestamp = earlier_timestamps.max()
                                    matching_rows = df[df['timestamp'] == closest_timestamp]
                            
                            if not matching_rows.empty:
                                # Update navigation position to the highlighted candlestick (using actual displayed timestamp)
                                # Find the position of the actual displayed timestamp in the current dataframe
                                actual_timestamp = matching_rows.iloc[0]['timestamp']
                                # Find where this actual timestamp appears in the current dataframe
                                position_rows = df[df['timestamp'] == actual_timestamp]
                                if not position_rows.empty:
                                    highlight_position = int(position_rows.index[0])
                                    st.session_state[nav_key] = highlight_position
                                    current_position = highlight_position
                                    navigation_was_updated = True
                            
                            # Update the previous highlights state
                            st.session_state[prev_highlights_key] = highlighted_timestamps.copy()
                    elif highlighted_timestamps and timeframe_changed:
                        # When timeframe changes and we have highlights, find the highlighted position in new timeframe
                        latest_highlight = max(highlighted_timestamps)
                        
                        # Find the closest matching row in the new timeframe dataframe
                        matching_rows = df[df['timestamp'] == latest_highlight]
                        if matching_rows.empty:
                            # Find closest earlier timestamp
                            earlier_timestamps = df[df['timestamp'] <= latest_highlight]['timestamp']
                            if not earlier_timestamps.empty:
                                closest_timestamp = earlier_timestamps.max()
                                matching_rows = df[df['timestamp'] == closest_timestamp]
                        
                        if not matching_rows.empty:
                            # Update navigation position to the highlighted candlestick in new timeframe
                            actual_timestamp = matching_rows.iloc[0]['timestamp']
                            position_rows = df[df['timestamp'] == actual_timestamp]
                            if not position_rows.empty:
                                highlight_position = position_rows.index[0]
                                st.session_state[nav_key] = highlight_position
                                current_position = highlight_position
                                navigation_was_updated = True
                    
                    # Always use navigation position when available, especially after updates
                    chart_pos = current_position  # Always use navigation position for consistent behavior
                    
                    fig = monitor.create_ohlc_chart(df, crypto, selected_timeframe, selected_indicator, indicator_params, highlighted_timestamps, chart_pos)
                    if fig:
                        # Show current highlights
                        if highlighted_timestamps:
                            highlight_info = []
                            for ts in highlighted_timestamps:
                                # Check if we have an exact match or need approximation
                                exact_match = df[df['timestamp'] == ts]
                                if not exact_match.empty:
                                    highlight_info.append(f"{ts.strftime('%H:%M:%S')}")
                                else:
                                    # Find closest earlier timestamp
                                    earlier_timestamps = df[df['timestamp'] <= ts]['timestamp']
                                    if not earlier_timestamps.empty:
                                        closest_timestamp = earlier_timestamps.max()
                                        highlight_info.append(f"{ts.strftime('%H:%M:%S')}→{closest_timestamp.strftime('%H:%M:%S')}")
                            st.info(f"💫 Currently highlighted: {sorted(highlight_info)}")
                            if any('→' in info for info in highlight_info):
                                st.caption("🔶 Orange stars show approximate matches (closest earlier time)")
                        
                        # Chart Navigation Controls - MOVED ABOVE CHART
                        st.markdown("**📊 Chart Navigation:**")
                        
                        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 2, 1])
                        
                        with nav_col1:
                            nav_back_disabled = bool(current_position <= 0)
                            if st.button("⬅️ Back", key=f"nav_back_{crypto}_{idx}", disabled=nav_back_disabled, use_container_width=True):
                                if current_position > 0:
                                    st.session_state[nav_key] = current_position - 1
                                    st.rerun()
                        
                        with nav_col2:
                            nav_forward_disabled = bool(current_position >= len(df) - 1)
                            if st.button("➡️ Forward", key=f"nav_forward_{crypto}_{idx}", disabled=nav_forward_disabled, use_container_width=True):
                                if current_position < len(df) - 1:
                                    st.session_state[nav_key] = current_position + 1
                                    st.rerun()
                        
                        with nav_col3:
                            # Show current position info
                            current_candle = df.iloc[current_position]
                            st.info(f"📍 Position: {current_position + 1}/{len(df)} | Time: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        with nav_col4:
                            nav_latest_disabled = bool(current_position >= len(df) - 1)
                            if st.button("🏠 Latest", key=f"nav_latest_{crypto}_{idx}", disabled=nav_latest_disabled, use_container_width=True):
                                st.session_state[nav_key] = len(df) - 1
                                st.rerun()
                        
                        # Use a stable chart key to prevent unnecessary recreation
                        chart_container = st.container()
                        with chart_container:
                            st.plotly_chart(
                                fig, 
                                use_container_width=True,
                                key=f"chart_{crypto}_{idx}_stable",
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
                        
                        # Highlight Controls - MOVED AFTER CHART
                        st.markdown("**🎯 Highlight Candlesticks:**")
                        
                        # Create a date/time picker for choosing candlestick to highlight
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        
                        with col1:
                            # Get date range from the data
                            min_date = df['timestamp'].min().date()
                            max_date = df['timestamp'].max().date()
                            min_time = df['timestamp'].min().time()
                            max_time = df['timestamp'].max().time()
                            
                            # Date picker
                            selected_date = st.date_input(
                                "Select Date:",
                                value=max_date,  # Default to most recent date
                                min_value=min_date,
                                max_value=max_date,
                                key=f"date_{crypto}_{idx}"
                            )
                            
                            # Time picker - get available times for selected date
                            available_times = df[df['timestamp'].dt.date == selected_date]['timestamp'].dt.time.tolist()
                            if available_times:
                                selected_time = st.selectbox(
                                    "Select Time:",
                                    options=available_times,
                                    index=len(available_times)-1,  # Default to most recent time
                                    format_func=lambda x: x.strftime('%H:%M:%S'),
                                    key=f"time_{crypto}_{idx}"
                                )
                                
                                # Find the corresponding timestamp
                                selected_datetime = pd.Timestamp.combine(selected_date, selected_time)
                                # Check if this timestamp exists in the data
                                matching_rows = df[df['timestamp'] == selected_datetime]
                                if not matching_rows.empty:
                                    st.info(f"Selected: {selected_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                                    selected_timestamp = selected_datetime
                                else:
                                    selected_timestamp = None
                                    st.warning("No candlestick found for selected date/time")
                            else:
                                selected_timestamp = None
                                st.warning("No data available for selected date")
                        
                        with col2:
                            if st.button("⭐ Highlight", key=f"highlight_{crypto}_{idx}"):
                                if selected_timestamp is not None and selected_timestamp not in highlighted_timestamps:
                                    highlighted_timestamps.append(selected_timestamp)
                                    st.session_state[highlight_key] = highlighted_timestamps
                                    st.rerun()
                        
                        with col3:
                            if st.button("❌ Remove", key=f"remove_{crypto}_{idx}"):
                                if selected_timestamp is not None and selected_timestamp in highlighted_timestamps:
                                    highlighted_timestamps.remove(selected_timestamp)
                                    st.session_state[highlight_key] = highlighted_timestamps
                                    # Update previous highlights state to reflect removal
                                    prev_highlights_key = f"prev_highlighted_candles_{crypto}"
                                    st.session_state[prev_highlights_key] = highlighted_timestamps.copy()
                                    st.rerun()
                        
                        with col4:
                            if st.button("🗑️ Clear All", key=f"clear_all_{crypto}_{idx}"):
                                if highlighted_timestamps:
                                    st.session_state[highlight_key] = []
                                    # Update previous highlights state to reflect clearing
                                    prev_highlights_key = f"prev_highlighted_candles_{crypto}"
                                    st.session_state[prev_highlights_key] = []
                                    st.rerun()
                        
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
            # Get highlighted candlesticks for this crypto (using timestamps)
            highlight_key = f"highlighted_candles_{crypto}"
            highlighted_timestamps = st.session_state.get(highlight_key, [])
            
            # Chart navigation state
            nav_key = f"chart_nav_{crypto}"
            timeframe_key = f"timeframe_{crypto}"
            
            # Check if timeframe has changed
            previous_timeframe = st.session_state.get(timeframe_key, None)
            current_timeframe = selected_timeframe
            timeframe_changed = previous_timeframe != current_timeframe
            
            if timeframe_changed:
                # Reset navigation to latest when timeframe changes
                st.session_state[nav_key] = len(df) - 1
                st.session_state[timeframe_key] = current_timeframe
            
            if nav_key not in st.session_state:
                st.session_state[nav_key] = len(df) - 1  # Start at the most recent candlestick
            
            current_position = int(st.session_state[nav_key])
            
            # Check if we should auto-pan to a new highlight
            should_auto_pan_to_highlight = False
            navigation_was_updated = False
            if highlighted_timestamps and not timeframe_changed:  # Don't auto-pan if timeframe just changed
                # Check if this is a new highlight by comparing with previous state
                prev_highlights_key = f"prev_highlighted_candles_{crypto}"
                prev_highlights = st.session_state.get(prev_highlights_key, [])
                
                # If we have new highlights, auto-pan to the latest one
                if len(highlighted_timestamps) > len(prev_highlights):
                    # Find the latest highlighted timestamp and set navigation position to it
                    latest_highlight = max(highlighted_timestamps)
                    
                    # Find the closest matching row in the dataframe (exact or closest earlier)
                    matching_rows = df[df['timestamp'] == latest_highlight]
                    if matching_rows.empty:
                        # Find closest earlier timestamp - this is what actually gets displayed
                        earlier_timestamps = df[df['timestamp'] <= latest_highlight]['timestamp']
                        if not earlier_timestamps.empty:
                            closest_timestamp = earlier_timestamps.max()
                            matching_rows = df[df['timestamp'] == closest_timestamp]
                    
                if not matching_rows.empty:
                    # Update navigation position to the highlighted candlestick (using actual displayed timestamp)
                    # Find the position of the actual displayed timestamp in the current dataframe
                    actual_timestamp = matching_rows.iloc[0]['timestamp']
                    # Find where this actual timestamp appears in the current dataframe
                    position_rows = df[df['timestamp'] == actual_timestamp]
                    if not position_rows.empty:
                        highlight_position = int(position_rows.index[0])
                        st.session_state[nav_key] = highlight_position
                        current_position = highlight_position
                        navigation_was_updated = True                    # Update the previous highlights state
                    st.session_state[prev_highlights_key] = highlighted_timestamps.copy()
                else:
                    st.write("📍 Using navigation position")
            elif highlighted_timestamps and timeframe_changed:
                # When timeframe changes and we have highlights, find the highlighted position in new timeframe
                latest_highlight = max(highlighted_timestamps)
                
                # Find the closest matching row in the new timeframe dataframe
                matching_rows = df[df['timestamp'] == latest_highlight]
                if matching_rows.empty:
                    # Find closest earlier timestamp
                    earlier_timestamps = df[df['timestamp'] <= latest_highlight]['timestamp']
                    if not earlier_timestamps.empty:
                        closest_timestamp = earlier_timestamps.max()
                        matching_rows = df[df['timestamp'] == closest_timestamp]
                
                if not matching_rows.empty:
                    # Update navigation position to the highlighted candlestick in new timeframe
                    actual_timestamp = matching_rows.iloc[0]['timestamp']
                    position_rows = df[df['timestamp'] == actual_timestamp]
                    if not position_rows.empty:
                        highlight_position = int(position_rows.index[0])
                        st.session_state[nav_key] = highlight_position
                        current_position = highlight_position
                        navigation_was_updated = True
            
            # Always use navigation position when available, especially after updates
            chart_pos = current_position  # Always use navigation position for consistent behavior
            
            fig = monitor.create_ohlc_chart(df, crypto, selected_timeframe, selected_indicator, indicator_params, highlighted_timestamps, chart_pos)
            if fig:
                # Show current highlights
                if highlighted_timestamps:
                    highlight_info = []
                    for ts in highlighted_timestamps:
                        # Check if we have an exact match or need approximation
                        exact_match = df[df['timestamp'] == ts]
                        if not exact_match.empty:
                            highlight_info.append(f"{ts.strftime('%H:%M:%S')}")
                        else:
                            # Find closest earlier timestamp
                            earlier_timestamps = df[df['timestamp'] <= ts]['timestamp']
                            if not earlier_timestamps.empty:
                                closest_timestamp = earlier_timestamps.max()
                                highlight_info.append(f"{ts.strftime('%H:%M:%S')}→{closest_timestamp.strftime('%H:%M:%S')}")
                    st.info(f"💫 Currently highlighted: {sorted(highlight_info)}")
                    if any('→' in info for info in highlight_info):
                        st.caption("🔶 Orange stars show approximate matches (closest earlier time)")
                
                # Chart Navigation Controls - MOVED ABOVE CHART
                st.markdown("### 📊 Chart Navigation")
                
                nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 2, 1])
                
                with nav_col1:
                    nav_back_disabled = bool(current_position <= 0)
                    if st.button("⬅️ Back", key=f"nav_back_{crypto}", disabled=nav_back_disabled, use_container_width=True):
                        if current_position > 0:
                            st.session_state[nav_key] = current_position - 1
                            st.rerun()
                
                with nav_col2:
                    nav_forward_disabled = bool(current_position >= len(df) - 1)
                    if st.button("➡️ Forward", key=f"nav_forward_{crypto}", disabled=nav_forward_disabled, use_container_width=True):
                        if current_position < len(df) - 1:
                            st.session_state[nav_key] = current_position + 1
                            st.rerun()
                
                with nav_col3:
                    # Show current position info
                    current_candle = df.iloc[current_position]
                    st.info(f"📍 Position: {current_position + 1}/{len(df)} | Time: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                with nav_col4:
                    nav_latest_disabled = bool(current_position >= len(df) - 1)
                    if st.button("🏠 Latest", key=f"nav_latest_{crypto}", disabled=nav_latest_disabled, use_container_width=True):
                        st.session_state[nav_key] = len(df) - 1
                        st.rerun()
                
                # Use a stable chart key to prevent unnecessary recreation
                chart_container = st.container()
                with chart_container:
                    st.plotly_chart(
                        fig, 
                        use_container_width=True,
                        key=f"chart_{crypto}_stable",
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
                
                # Highlighting interface
                st.markdown("### 🎯 Highlight Candlesticks")
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    # Get date range from the data
                    min_date = df['timestamp'].min().date()
                    max_date = df['timestamp'].max().date()
                    
                    # Date picker
                    selected_date = st.date_input(
                        "Select Date:",
                        value=max_date,  # Default to most recent date
                        min_value=min_date,
                        max_value=max_date,
                        key=f"date_single_{crypto}"
                    )
                    
                    # Time picker - get available times for selected date
                    available_times = df[df['timestamp'].dt.date == selected_date]['timestamp'].dt.time.tolist()
                    if available_times:
                        selected_time = st.selectbox(
                            "Select Time:",
                            options=available_times,
                            index=len(available_times)-1,  # Default to most recent time
                            format_func=lambda x: x.strftime('%H:%M:%S'),
                            key=f"time_single_{crypto}"
                        )
                        
                        # Find the corresponding timestamp
                        selected_datetime = pd.Timestamp.combine(selected_date, selected_time)
                        # Check if this timestamp exists in the data
                        matching_rows = df[df['timestamp'] == selected_datetime]
                        if not matching_rows.empty:
                            st.info(f"Selected: {selected_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                            selected_timestamp = selected_datetime
                        else:
                            selected_timestamp = None
                            st.warning("No candlestick found for selected date/time")
                    else:
                        selected_timestamp = None
                        st.warning("No data available for selected date")
                
                with col2:
                    if st.button("⭐ Highlight", key=f"highlight_btn_{crypto}"):
                        if selected_timestamp is not None:
                            if f"highlighted_candles_{crypto}" not in st.session_state:
                                st.session_state[f"highlighted_candles_{crypto}"] = []
                            if selected_timestamp not in st.session_state[f"highlighted_candles_{crypto}"]:
                                st.session_state[f"highlighted_candles_{crypto}"].append(selected_timestamp)
                                st.rerun()
                
                with col3:
                    if st.button("🗑️ Remove", key=f"remove_btn_{crypto}"):
                        if selected_timestamp is not None:
                            if f"highlighted_candles_{crypto}" in st.session_state:
                                if selected_timestamp in st.session_state[f"highlighted_candles_{crypto}"]:
                                    st.session_state[f"highlighted_candles_{crypto}"].remove(selected_timestamp)
                                    # Update previous highlights state to reflect removal
                                    prev_highlights_key = f"prev_highlighted_candles_{crypto}"
                                    st.session_state[prev_highlights_key] = st.session_state[f"highlighted_candles_{crypto}"].copy()
                                    st.rerun()
                
                with col4:
                    if st.button("🗑️ Clear All", key=f"clear_all_btn_{crypto}"):
                        if f"highlighted_candles_{crypto}" in st.session_state:
                            if st.session_state[f"highlighted_candles_{crypto}"]:
                                st.session_state[f"highlighted_candles_{crypto}"] = []
                                # Update previous highlights state to reflect clearing
                                prev_highlights_key = f"prev_highlighted_candles_{crypto}"
                                st.session_state[prev_highlights_key] = []
                                st.rerun()
                
                # Display current highlights
                if f"highlighted_candles_{crypto}" in st.session_state and st.session_state[f"highlighted_candles_{crypto}"]:
                    highlighted_info = []
                    for timestamp in st.session_state[f"highlighted_candles_{crypto}"]:
                        # Find the corresponding row in current data
                        matching_rows = df[df['timestamp'] == timestamp]
                        if not matching_rows.empty:
                            candle_data = matching_rows.iloc[0]
                            highlighted_info.append(
                                f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} - Close: ${candle_data['close']:.2f}"
                            )
                        else:
                            highlighted_info.append(f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} - (Data not available)")
                    
                    st.info(f"Currently highlighted: {', '.join(highlighted_info)}")
                
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
    
    # Remove auto-refresh to prevent scroll jumping
    # Auto-refresh functionality (disabled to prevent scroll issues)
    # time.sleep(AUTO_REFRESH_INTERVAL)
    # st.rerun()

if __name__ == "__main__":
    main()
