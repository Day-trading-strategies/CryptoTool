import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import ta
import requests
import time as time_module
from datetime import datetime, timedelta, date, time
import ccxt
from config import *

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

    def compute_indicators(self, df, indicators, params):
        """Add all requested indicator columns to df in one pass."""
        if "RSI" in indicators:
            df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=params["RSI"]["window"]).rsi()
        if "William % Range" in indicators:
            df["WR"] = ta.momentum.williams_r(df["high"], df["low"], df["close"], lbp=params["William % Range"]["lbp"])
        if "Bollinger Band" in indicators:
            bb = ta.volatility.BollingerBands(df["close"], window=params["Bollinger Band"]["window"],
                                              window_dev=params["Bollinger Band"]["window_dev"])
            df["bb_middle"], df["bb_upper"], df["bb_lower"] = bb.bollinger_mavg(), bb.bollinger_hband(), bb.bollinger_lband()
        if "KDJ" in indicators:
            stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"],
                                                     window=params["KDJ"]["period"],
                                                     smooth_window=params["KDJ"]["signal"])
            df["%K"], df["%D"] = stoch.stoch(), stoch.stoch_signal()
            df["%J"] = 3 * df["%K"] - 2 * df["%D"]
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

class CryptoBacktester:
    def __init__(self, monitor, start_date, end_date, conditions:dict):
        self.monitor = monitor
        self.start_date = datetime.combine(start_date, time.min)
        self.end_date   = datetime.combine(end_date, time.max)
        self.indicator_conditions = conditions
        self.tf_to_ms = {
            '1m':  60_000,
            '3m':  3 * 60_000,
            '5m':  5 * 60_000,
            '15m': 15 * 60_000,
            '1h':  60 * 60_000,
            '4h':  4 * 60 * 60_000,
            '1d':  24 * 60 * 60_000,
        }
    # grabs data from start_date to end_date
    def fetch_historical(self, symbol, timeframe):
        """
        Fetch all OHLC bars from start_date up to end_date,
        without explicitly setting a `limit`—we rely on CCXT's default.
        """
        all_bars = []
        # turns dates into timestamps
        since_ms = int(self.start_date.timestamp() * 1_000)
        end_ms   = int(self.end_date.timestamp()   * 1_000)
        step     = self.tf_to_ms[timeframe]
        while since_ms <= end_ms:
            # no `limit` argument here → CCXT/binance uses its default (500)
            ohlcv = self.monitor.exchange.fetch_ohlcv(
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
            # stop if we've already covered end_ms
            if last_ts >= end_ms:
                break
            # advance since to one bar past the last timestamp
            since_ms = last_ts + step
            # if we got fewer than the default page-size (500), we're done
            if len(ohlcv) < 500:
                break
        # build DataFrame
        df = pd.DataFrame(all_bars, columns=[
            'timestamp','open','high','low','close','volume'
        ])
        df.drop('volume', axis=1, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df


    # Will return timestamps. Maybe return values of indicators as well?
    def find_hits(self, df: pd.DataFrame, epsilon: float = 0.5) -> pd.DataFrame:
        # building mask knowing all rows have real numbers
        mask = pd.Series(True, index=df.index)
        up = pd.Series(False, index=df.index)
        down = pd.Series(False, index=df.index)
        touch_top = pd.Series(False, index=df.index)
        touch_bot = pd.Series(False, index=df.index)
        if "RSI_U" in self.indicator_conditions or "RSI_L" in self.indicator_conditions:
            up = pd.Series(False, index=df.index)
            down = pd.Series(False, index=df.index)
            if "RSI_U" in self.indicator_conditions:
                thr = self.indicator_conditions["RSI_U"]
                rsi = df["rsi"]
                up   = (rsi.shift(1) < thr) & (rsi >= thr)
            if "RSI_L" in self.indicator_conditions:
                thr = self.indicator_conditions["RSI_L"]
                rsi = df["rsi"]
                down = (rsi.shift(1) > thr) & (rsi <= thr)
            mask &= (up | down)
        # William %R crossing
        if "William % Range" in self.indicator_conditions:
            thr = self.indicator_conditions["William % Range"]
            wr  = df["WR"]
            up   = (wr.shift(1) < thr) & (wr >= thr)
            down = (wr.shift(1) > thr) & (wr <= thr)
            mask &= (up | down)
        # Bollinger Band touches
        if ("Bollinger Top" in self.indicator_conditions or
            "Bollinger Bottom" in self.indicator_conditions or
            "Bollinger Either" in self.indicator_conditions
            ):
            # start with both False
            touch_top = pd.Series(False, index=df.index)
            touch_bot = pd.Series(False, index=df.index)
            up = df["bb_upper"]
            touch_top = (
                (df["high"].shift(1) < up.shift(1))
                & (df["high"] >= up)
            )
            low = df["bb_lower"]
            touch_bot = (
                (df["low"].shift(1) > low.shift(1))
                & (df["low"] <= low)
            )
            # top‐band touch?
            if "Bollinger Top" in self.indicator_conditions:
                mask &= touch_top
            # bottom‐band touch?
            if "Bollinger Bottom" in self.indicator_conditions:
                mask &= touch_bot

            if "Bollinger Either" in self.indicator_conditions:
                mask &= (touch_top | touch_bot)
            # only now AND in whichever the user asked for
        # KDJ intersection
        if "KDJ" in self.indicator_conditions:
            K, D, J = df["%K"], df["%D"], df["%J"]
            intersect = (K.sub(D).abs() < epsilon) & (K.sub(J).abs() < epsilon)
            mask &= intersect
        # Return only the timestamps that made it through all tests
        return df.loc[mask].reset_index(drop=True)

def main():
    """Main function to run the Streamlit app"""
    
    # Initialize session state variables for both modes
    if 'monitor_mode' not in st.session_state:
        st.session_state.monitor_mode = 'live'  # 'live' or 'practice'
    
    # Initialize highlighted_timestamps for live mode
    if 'highlighted_timestamps' not in st.session_state:
        st.session_state.highlighted_timestamps = []
    
    # Initialize navigation state for live mode
    if 'chart_position' not in st.session_state:
        st.session_state.chart_position = None
    
    # Initialize practice mode state
    if 'practice_results' not in st.session_state:
        st.session_state.practice_results = None
    if 'practice_symbol' not in st.session_state:
        st.session_state.practice_symbol = None
    if 'practice_timeframe' not in st.session_state:
        st.session_state.practice_timeframe = None
    if 'practice_df' not in st.session_state:
        st.session_state.practice_df = None
    
    st.title("🚀 Crypto Price Monitor & Backtester")
    
    # Mode selection
    mode = st.selectbox(
        "Select Mode:",
        ["Live Monitoring", "Practice/Backtest"],
        index=0 if st.session_state.monitor_mode == 'live' else 1
    )
    
    # Update session state
    st.session_state.monitor_mode = 'live' if mode == "Live Monitoring" else 'practice'
    
    # Initialize the monitor
    monitor = CryptoPriceMonitor()
    
    if st.session_state.monitor_mode == 'live':
        # Live Monitoring Mode
        
        # Create sidebar for settings
        with st.sidebar:
            st.header("📊 Chart Settings")
            
            # Select multiple cryptocurrencies
            selected_cryptos = st.multiselect(
                "Select Cryptocurrencies:",
                options=list(monitor.available_cryptos.keys()),
                default=DEFAULT_CRYPTOS
            )
            
            # Select timeframe
            timeframe = st.selectbox(
                "Select Timeframe:",
                options=list(monitor.timeframes.keys()),
                index=list(monitor.timeframes.keys()).index(DEFAULT_TIMEFRAME)
            )
            
            # Select indicators to display
            st.subheader("Technical Indicators")
            indicators = st.multiselect(
                "Select Indicators:",
                options=["Half Trend", "RSI", "William % Range", "Bollinger Band", "KDJ"],
                default=DEFAULT_INDICATORS
            )
            
            # Indicator parameters
            params = {}
            if "Half Trend" in indicators:
                st.subheader("Half Trend Settings")
                params['Half Trend'] = {
                    'period': st.slider("Period", 1, 50, HALF_TREND_PERIOD),
                    'multiplier': st.slider("Multiplier", 0.1, 5.0, HALF_TREND_MULTIPLIER, 0.1)
                }
            
            if "RSI" in indicators:
                st.subheader("RSI Settings")
                params['RSI'] = {
                    'window': st.slider("RSI Window", 1, 50, RSI_WINDOW)
                }
            
            if "William % Range" in indicators:
                st.subheader("William % Range Settings")
                params['William % Range'] = {
                    'lbp': st.slider("Lookback Period", 1, 50, WR_LBP)
                }
            
            if "Bollinger Band" in indicators:
                st.subheader("Bollinger Bands Settings")
                params['Bollinger Band'] = {
                    'window': st.slider("BB Window", 1, 50, BB_WINDOW),
                    'window_dev': st.slider("BB Std Dev", 0.1, 5.0, BB_WINDOW_DEV, 0.1)
                }
            
            if "KDJ" in indicators:
                st.subheader("KDJ Settings")
                params['KDJ'] = {
                    'period': st.slider("KDJ Period", 1, 50, KDJ_PERIOD),
                    'signal': st.slider("KDJ Signal", 1, 10, KDJ_SIGNAL)
                }
            
            # Refresh button
            if st.button("🔄 Refresh Data", type="primary"):
                st.cache_data.clear()
        
        # Main content area for live mode
        if selected_cryptos:
            # Display price summary
            st.subheader("💰 Current Prices")
            monitor.display_price_summary(selected_cryptos, timeframe)
            
            # Highlight Controls
            st.subheader("🎯 Highlight Controls")
            
            # Create columns for highlight controls
            h_col1, h_col2, h_col3, h_col4 = st.columns([3, 2, 2, 2])
            
            with h_col1:
                # Input for adding highlights
                highlight_input = st.text_input(
                    "Add Highlight (YYYY-MM-DD HH:MM:SS):",
                    placeholder="2024-01-15 14:30:00",
                    help="Enter timestamp to highlight on chart"
                )
            
            with h_col2:
                if st.button("➕ Add Highlight", type="secondary"):
                    try:
                        # Parse the input timestamp
                        highlight_time = datetime.strptime(highlight_input, "%Y-%m-%d %H:%M:%S")
                        
                        # Add to session state
                        if highlight_time not in st.session_state.highlighted_timestamps:
                            st.session_state.highlighted_timestamps.append(highlight_time)
                            st.rerun()
                        else:
                            st.warning("Timestamp already highlighted!")
                    except ValueError:
                        if highlight_input.strip():  # Only show error if there was input
                            st.error("Invalid format! Use YYYY-MM-DD HH:MM:SS")
            
            with h_col3:
                if st.button("🗑️ Clear All", type="secondary"):
                    st.session_state.highlighted_timestamps = []
                    st.rerun()
            
            with h_col4:
                if st.button("📍 Go to Latest", type="secondary"):
                    # Reset navigation to show latest data
                    st.session_state.chart_position = None
                    st.rerun()
            
            # Display current highlights
            if st.session_state.highlighted_timestamps:
                st.write("**Current Highlights:**")
                for i, ts in enumerate(st.session_state.highlighted_timestamps):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"• {ts.strftime('%Y-%m-%d %H:%M:%S')}")
                    with col2:
                        if st.button("❌", key=f"remove_{i}", help="Remove this highlight"):
                            st.session_state.highlighted_timestamps.pop(i)
                            st.rerun()
            
            # Charts section
            st.subheader("📈 Live Charts")
            
            # Fetch and display charts for each selected crypto
            for crypto in selected_cryptos:
                symbol = monitor.available_cryptos[crypto]
                
                # Create columns for chart and navigation
                chart_col, nav_col = st.columns([4, 1])
                
                with chart_col:
                    with st.spinner(f'Loading {crypto} chart...'):
                        df = monitor.fetch_ohlc_data(symbol, monitor.timeframes[timeframe], limit=500)
                        
                        if df is not None and not df.empty:
                            # Compute indicators for the chart
                            df = monitor.compute_indicators(df, indicators, params)
                            
                            # Create chart with highlights and navigation position
                            fig = monitor.create_ohlc_chart(
                                df, crypto, timeframe, indicators, params,
                                highlighted_timestamps=st.session_state.highlighted_timestamps,
                                chart_position=st.session_state.chart_position
                            )
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error(f"Unable to create chart for {crypto}")
                        else:
                            st.error(f"Unable to fetch data for {crypto}")
                
                with nav_col:
                    st.write("**Navigation**")
                    
                    # Navigation controls for this specific chart
                    nav_key = f"nav_{crypto}"  # Unique key for each crypto
                    
                    if st.button("⏮️ Start", key=f"{nav_key}_start", help="Go to chart start"):
                        if df is not None and not df.empty:
                            st.session_state.chart_position = len(df) - 1  # Start = oldest data (end of df)
                            st.rerun()
                    
                    if st.button("⏪ <<", key=f"{nav_key}_back10", help="Go back 10 candles"):
                        if df is not None and not df.empty:
                            current_pos = st.session_state.chart_position or 0
                            new_pos = min(current_pos + 10, len(df) - 1)
                            st.session_state.chart_position = new_pos
                            st.rerun()
                    
                    if st.button("◀️ <", key=f"{nav_key}_back", help="Go back 1 candle"):
                        if df is not None and not df.empty:
                            current_pos = st.session_state.chart_position or 0
                            new_pos = min(current_pos + 1, len(df) - 1)
                            st.session_state.chart_position = new_pos
                            st.rerun()
                    
                    if st.button("▶️ >", key=f"{nav_key}_forward", help="Go forward 1 candle"):
                        if df is not None and not df.empty:
                            current_pos = st.session_state.chart_position
                            if current_pos is not None and current_pos > 0:
                                st.session_state.chart_position = current_pos - 1
                                st.rerun()
                    
                    if st.button("⏩ >>", key=f"{nav_key}_forward10", help="Go forward 10 candles"):
                        if df is not None and not df.empty:
                            current_pos = st.session_state.chart_position
                            if current_pos is not None:
                                new_pos = max(current_pos - 10, 0)
                                st.session_state.chart_position = new_pos if new_pos > 0 else None
                                st.rerun()
                    
                    if st.button("⏭️ Latest", key=f"{nav_key}_latest", help="Go to latest data"):
                        st.session_state.chart_position = None
                        st.rerun()
                    
                    # Show current position
                    if df is not None and not df.empty:
                        if st.session_state.chart_position is not None:
                            pos = st.session_state.chart_position
                            if 0 <= pos < len(df):
                                current_time = df.iloc[pos]['timestamp']
                                st.write(f"**Position:** {len(df) - pos}/{len(df)}")
                                st.write(f"**Time:** {current_time.strftime('%m/%d %H:%M')}")
                        else:
                            st.write("**Position:** Latest")
                
                st.markdown("---")  # Separator between charts
        
        else:
            st.warning("Please select at least one cryptocurrency to display.")
    
    else:
        # Practice/Backtest Mode
        st.header("🎯 Practice Mode - Historical Backtesting")
        
        # Backtest configuration
        col1, col2 = st.columns(2)
        
        with col1:
            # Symbol selection
            practice_crypto = st.selectbox(
                "Select Cryptocurrency for Backtesting:",
                options=list(monitor.available_cryptos.keys()),
                index=0
            )
            
            # Date range
            start_date = st.date_input(
                "Start Date",
                value=date.today() - timedelta(days=30),
                max_value=date.today()
            )
            
            end_date = st.date_input(
                "End Date",
                value=date.today(),
                max_value=date.today()
            )
        
        with col2:
            # Timeframe selection
            practice_timeframe = st.selectbox(
                "Select Timeframe for Backtesting:",
                options=list(monitor.timeframes.keys()),
                index=2  # Default to 5m
            )
            
            # Indicator conditions
            st.subheader("Indicator Conditions")
            
            conditions = {}
            
            # RSI conditions
            if st.checkbox("RSI Upper Threshold"):
                conditions["RSI_U"] = st.slider("RSI Upper", 50, 100, 70)
            
            if st.checkbox("RSI Lower Threshold"):
                conditions["RSI_L"] = st.slider("RSI Lower", 0, 50, 30)
            
            # William %R conditions
            if st.checkbox("William %R Threshold"):
                conditions["William % Range"] = st.slider("William %R", -100, 0, -20)
            
            # Bollinger Band conditions
            bb_option = st.selectbox(
                "Bollinger Band Touch:",
                ["None", "Top Band", "Bottom Band", "Either Band"]
            )
            
            if bb_option == "Top Band":
                conditions["Bollinger Top"] = True
            elif bb_option == "Bottom Band":
                conditions["Bollinger Bottom"] = True
            elif bb_option == "Either Band":
                conditions["Bollinger Either"] = True
            
            # KDJ intersection
            if st.checkbox("KDJ Line Intersection"):
                conditions["KDJ"] = True
        
        # Run backtest button
        if st.button("🚀 Run Backtest", type="primary"):
            if conditions:  # Only run if at least one condition is selected
                with st.spinner("Running backtest..."):
                    try:
                        # Create backtester
                        symbol = monitor.available_cryptos[practice_crypto]
                        backtester = CryptoBacktester(monitor, start_date, end_date, conditions)
                        
                        # Fetch historical data
                        historical_df = backtester.fetch_historical(symbol, monitor.timeframes[practice_timeframe])
                        
                        if historical_df is not None and not historical_df.empty:
                            # Set up indicators needed for conditions
                            indicators_needed = []
                            params_needed = {}
                            
                            if "RSI_U" in conditions or "RSI_L" in conditions:
                                indicators_needed.append("RSI")
                                params_needed["RSI"] = {"window": RSI_WINDOW}
                            
                            if "William % Range" in conditions:
                                indicators_needed.append("William % Range")
                                params_needed["William % Range"] = {"lbp": WR_LBP}
                            
                            if any(key in conditions for key in ["Bollinger Top", "Bollinger Bottom", "Bollinger Either"]):
                                indicators_needed.append("Bollinger Band")
                                params_needed["Bollinger Band"] = {"window": BB_WINDOW, "window_dev": BB_WINDOW_DEV}
                            
                            if "KDJ" in conditions:
                                indicators_needed.append("KDJ")
                                params_needed["KDJ"] = {"period": KDJ_PERIOD, "signal": KDJ_SIGNAL}
                            
                            # Compute indicators
                            historical_df = monitor.compute_indicators(historical_df, indicators_needed, params_needed)
                            
                            # Find hits
                            hits_df = backtester.find_hits(historical_df)
                            
                            # Store results in session state
                            st.session_state.practice_results = hits_df
                            st.session_state.practice_symbol = practice_crypto
                            st.session_state.practice_timeframe = practice_timeframe
                            st.session_state.practice_df = historical_df
                            
                            st.success(f"Backtest completed! Found {len(hits_df)} hits.")
                        
                        else:
                            st.error("Unable to fetch historical data for the selected period.")
                    
                    except Exception as e:
                        st.error(f"Error running backtest: {str(e)}")
            else:
                st.warning("Please select at least one indicator condition.")
        
        # Display results
        if st.session_state.practice_results is not None:
            st.subheader("📊 Backtest Results")
            
            hits_df = st.session_state.practice_results
            
            if not hits_df.empty:
                st.write(f"**Total Hits Found:** {len(hits_df)}")
                
                # Convert hits to timestamps for highlighting
                hit_timestamps = hits_df['timestamp'].tolist()
                
                # Display hit timestamps
                st.write("**Hit Timestamps:**")
                for i, hit_time in enumerate(hit_timestamps[:10]):  # Show first 10
                    st.write(f"• {hit_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if len(hit_timestamps) > 10:
                    st.write(f"... and {len(hit_timestamps) - 10} more")
                
                # Create chart with hits highlighted
                if st.session_state.practice_df is not None:
                    st.subheader("📈 Backtest Chart")
                    
                    # Set up all indicators for chart display
                    all_indicators = ["RSI", "William % Range", "Bollinger Band", "KDJ"]
                    all_params = {
                        "RSI": {"window": RSI_WINDOW},
                        "William % Range": {"lbp": WR_LBP},
                        "Bollinger Band": {"window": BB_WINDOW, "window_dev": BB_WINDOW_DEV},
                        "KDJ": {"period": KDJ_PERIOD, "signal": KDJ_SIGNAL}
                    }
                    
                    # Compute all indicators for display
                    chart_df = monitor.compute_indicators(st.session_state.practice_df, all_indicators, all_params)
                    
                    # Create chart with hits highlighted
                    fig = monitor.create_ohlc_chart(
                        chart_df,
                        st.session_state.practice_symbol,
                        st.session_state.practice_timeframe,
                        all_indicators,
                        all_params,
                        highlighted_timestamps=hit_timestamps
                    )
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Unable to create backtest chart")
            
            else:
                st.warning("No hits found with the selected conditions. Try adjusting the thresholds.")

if __name__ == "__main__":
    main()
