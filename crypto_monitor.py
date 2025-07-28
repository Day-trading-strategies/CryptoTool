import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        weights = [3] + [1] * len(sep_inds)
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

class CryptoBacktester:
    def __init__(self, monitor, start_date, end_date, conditions:dict):
        self.monitor = monitor
        self.start_date = datetime.combine(start_date, time.min)
        self.end_date   = datetime.combine(end_date, time.max)
        self.indicator_conditions = conditions
        self.tf_to_ms = {
            '1m':  60_000,
            '3m':  3 * 60_000
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
        without explicitly setting a `limit`—we rely on CCXT’s default.
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
    st.title("🚀 Crypto Price Monitor")
    st.markdown("---")        

    # Initialize the monitor
    monitor = CryptoPriceMonitor()
    
    # Initialize session_state for backtest persistence ---
    for key in ("bt_mode","bt_df","bt_crypto"):
        if key not in st.session_state:
            st.session_state[key] = None
    if "chart_end" not in st.session_state:
        st.session_state.chart_end = None

    if st.session_state.bt_mode is None:
        st.session_state.bt_mode = False
    
    if "hit_index" not in st.session_state:
        st.session_state.hit_index = 0  # “no hit selected” state 

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
        index=6  # Default to 1h
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


    if st.session_state.bt_mode:
        symbol = monitor.available_cryptos[st.session_state["bt_crypto"]]
        crypto = st.session_state["bt_crypto"]

        st.session_state["bt_df"] = st.session_state["ob"].fetch_historical(symbol, selected_timeframe)
        df = st.session_state["bt_df"]
        fig = monitor.create_ohlc_chart(st.session_state["bt_df"], crypto, selected_timeframe, selected_indicator, indicator_params)

        # if user chooses a hit point to test, create chart from start to hit point
        if st.session_state.chart_end:
            temp_df = df[df["timestamp"] <= st.session_state.chart_end]
            fig = monitor.create_ohlc_chart(temp_df, crypto, selected_timeframe, selected_indicator, indicator_params)
        # else create chart with original data.

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
            # create csv file with all the prices and indicators
            st.session_state["bt_df"].to_csv("backtest_data.csv", index=False)

            hits = st.session_state["ob"].find_hits(st.session_state["bt_df"])
            timestamps = hits["timestamp"].tolist()
            # we only need the timestamps for click‐points
            hits["timestamp"].to_csv("filtered_backtest.csv", index=False) 

            # render a “Next →” button
            if st.button("Next →", key="next_hit"):
                # advance index (wrap around)
                if st.session_state.hit_index < len(timestamps) - 1:
                    st.session_state.hit_index += 1
                else:
                    st.session_state.hit_index = 0
                # set chart_end to the newly selected hit
                st.session_state.chart_end = timestamps[st.session_state.hit_index]

                      
            try:
                filtered = pd.read_csv("filtered_backtest.csv")
                filtered["timestamp"] = pd.to_datetime(filtered["timestamp"])
                # turn timestamps into strings for display
                times = filtered["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
            except FileNotFoundError:
                times = []

            # Show them in an expander as clickable buttons
            with st.expander(f"Found {len(timestamps)} hit points."):
                if not times:
                    st.write("No hits found (or filtered_backtest.csv not present).")
                else:
                    for idx, ts in enumerate(times):
                        # each timestamp is a button; clicking does nothing for now
                        if st.button(ts, key=f"hit_btn_{idx}"):
                            # parse back into a datetime
                            st.session_state.chart_end = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                            st.rerun()

        else:
            st.error(f"Unable to Backtest Chart")
            

    # Create tabs for each cryptocurrency
    elif len(selected_cryptos) > 1:
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

    with st.expander("Backtesting Settings"):
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=30),
            key="bt_start"
        )
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date(),
            key="bt_end"
        )
        #number
        st.session_state["bt_crypto"] = st.selectbox(
        "Select Crypto:",
        options=list(monitor.available_cryptos.keys()),
        index=0  # Default to BTC
        )
        st.markdown("**Indicator Conditions(select indicator in sidebar to show.)**")
        
        indicator_conditions = {}
        if "RSI" in selected_indicator:
            # RSI
            if st.checkbox("RSI Upper Bound", key="bt_chk_rsi_u"):#number
                indicator_conditions["RSI_U"] = st.number_input("RSI Upper Bound", 0.0, 100.0, 70.0, step=0.1, key="bt_rsi_u")
            if st.checkbox("RSI Lower Bound", key="bt_chk_rsi_l"):
                indicator_conditions["RSI_L"] = st.number_input("RSI Lower Bound", 0.0, 100.0, 30.0, step=0.1, key="bt_rsi_l")

        if "Bollinger Band" in selected_indicator:
            # Bollinger Band
            if st.checkbox("BB Price Touches Top Band"):
                indicator_conditions["Bollinger Top"] = True
            if st.checkbox("BB Price Touches Bottom Band"):
                indicator_conditions["Bollinger Bottom"] = True
            if st.checkbox("BB Touch Either"):
                indicator_conditions["Bollinger Either"] = True

        if "William % Range" in selected_indicator:
            # William %R
            if st.checkbox("William %R Condition", key="bt_chk_wr"):
                indicator_conditions["William % Range"] = st.number_input(
                    "William %R Threshold (e.g. -20)", -100.0, 0.0, -20.0, step=0.1, key="bt_wr"
                )
        if "KDJ" in selected_indicator:
            # KDJ
            if st.checkbox("KDJ Intersection", key="bt_chk_kdj"):
                indicator_conditions["KDJ"] = True
        if "Half Trend" in selected_indicator:
            # Half Trend
            if st.checkbox("Half Trend Condition", key="bt_chk_ht"):
                indicator_conditions["Half Trend"] = st.number_input(
                    "Half Trend Multiplier Threshold (e.g. 1.0)", 0.1, 5.0, 1.0, step=0.1, key="bt_ht"
                )

        run_bt = st.button("▶ Run Backtest", key="bt_run")

    # backtesting 
    # session_state variables will save [object, data]
    if run_bt:
        st.session_state.bt_mode = True
        st.session_state["ob"] = CryptoBacktester(monitor, start_date, end_date, indicator_conditions)
        st.rerun()
        
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
    if not st.session_state.bt_mode:
        time_module.sleep(AUTO_REFRESH_INTERVAL)
        st.rerun()

if __name__ == "__main__":
    main()
