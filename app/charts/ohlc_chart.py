import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pytz
import ta

from app.config import *

from app.indicators.half_trend import HalfTrendIndicator
from app.indicators.bollinger_bands import BollingerBandsIndicator
from app.indicators.rsi import RSIIndicator
from app.indicators.williams_r import WilliamsRIndicator
from app.indicators.kdj import KDJIndicator

class OHLCChartCreator:
    """Class to create OHLC charts with optional indicators and highlighting functionality."""
    
    def __init__(self, selected_cryptos, timeframe, data_fetcher, selected_indicators, indicator_params):
        self.df = {crypto: [] for crypto in selected_cryptos}
        self.selected_cryptos = selected_cryptos
        self.selected_indicators = selected_indicators if selected_indicators is not None else []
        self.indicator_params = indicator_params
        self.timeframe = timeframe
        self.data_fetcher = data_fetcher
        self.separate_ax_indicators = SEPARATE_AX_INDICATORS
        self.render()

    def render(self):
        """Render charts with highlighting and navigation functionality exactly like the original"""
        st.subheader("📈 OHLC Charts")
        
        # Create tabs for each cryptocurrency
        if len(self.selected_cryptos) > 1:
            tabs = st.tabs(self.selected_cryptos)
            for idx, crypto in enumerate(self.selected_cryptos):
                with tabs[idx]:
                    self._render_crypto_chart(crypto, idx)
        else:
            # Single cryptocurrency view
            crypto = self.selected_cryptos[0]
            self._render_crypto_chart(crypto, 0)

    def _render_crypto_chart(self, crypto, idx):
        """Render individual crypto chart with exact original functionality"""
        # Fetch data
        self.df[crypto] = self.data_fetcher.fetch_ohlc_data(AVAILABLE_CRYPTOS[crypto], self.timeframe)
        if self.df[crypto] is None:
            st.error(f"Unable to load chart for {crypto}")
            return
        
        df = self.df[crypto]
        
        # Get highlighted candlesticks for this crypto (using timestamps)
        highlight_key = f"highlighted_candles_{crypto}"
        highlighted_timestamps = st.session_state.get(highlight_key, [])
        
        # Chart navigation state
        nav_key = f"chart_nav_{crypto}"
        timeframe_key = f"timeframe_{crypto}"
        
        # Check if timeframe has changed
        previous_timeframe = st.session_state.get(timeframe_key, None)
        current_timeframe = self.timeframe
        timeframe_changed = previous_timeframe != current_timeframe
        
        if timeframe_changed:
            # Reset navigation to latest when timeframe changes
            st.session_state[nav_key] = len(df) - 1
            st.session_state[timeframe_key] = current_timeframe
        
        if nav_key not in st.session_state:
            st.session_state[nav_key] = len(df) - 1  # Start at the most recent candlestick
        
        current_position = int(st.session_state[nav_key])
        
        # Handle highlighting auto-pan logic
        current_position = self._handle_highlighting_navigation(
            crypto, highlighted_timestamps, df, current_position, 
            timeframe_changed, nav_key
        )
        
        # Show current highlights info if any
        if highlighted_timestamps:
            st.info(f"🎯 {len(highlighted_timestamps)} highlighted candlesticks")
        
        # Chart Navigation Buttons (without slider)
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            if st.button("⏮️ First", key=f"first_{crypto}_{idx}"):
                st.session_state[nav_key] = 0
                st.rerun()
        with col2:
            if st.button("⬅️ Prev", key=f"prev_{crypto}_{idx}"):
                st.session_state[nav_key] = max(0, current_position - 1)
                st.rerun()
        with col3:
            if st.button("➡️ Next", key=f"next_{crypto}_{idx}"):
                st.session_state[nav_key] = min(len(df) - 1, current_position + 1)
                st.rerun()
        with col4:
            if st.button("⏭️ Latest", key=f"latest_{crypto}_{idx}"):
                st.session_state[nav_key] = len(df) - 1
                st.rerun()
        
        # Show current position info
        st.caption(f"Position: {current_position + 1} of {len(df)}")
        
        # Create chart with exact original functionality
        fig = self.create_ohlc_chart(df, crypto, self.timeframe, self.selected_indicators, 
                                   self.indicator_params, highlighted_timestamps, current_position)
        
        if fig:
            # Display the chart with original config
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
            
            # Highlight Controls (exactly like original)
            self._render_highlight_controls(crypto, idx, df, highlight_key)
            
            # Display data table (exactly like original)
            with st.expander(f"📋 {crypto} Data Table"):
                st.dataframe(df.tail(10).iloc[::-1], use_container_width=True)

    def calculate_half_trend(self, df, period=10, multiplier=1):
        """Calculate Half Trend indicator exactly like original"""
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

    def convert_utc_to_local(self, timestamp_series):
        """Convert UTC timestamps to local time for display"""
        try:
            import time
            is_dst = time.daylight and time.localtime().tm_isdst > 0
            utc_offset_seconds = - (time.altzone if is_dst else time.timezone)
            local_tz_offset = timedelta(seconds=utc_offset_seconds)
            
            # Convert series of UTC timestamps to local time
            if hasattr(timestamp_series.iloc[0], 'tz') and timestamp_series.iloc[0].tz is not None:
                # Timezone-aware timestamps - convert to local
                return timestamp_series.dt.tz_convert(None) + local_tz_offset
            else:
                # Timezone-naive timestamps (assumed UTC) - add offset
                return timestamp_series + local_tz_offset
        except:
            # Fallback: return original timestamps if conversion fails
            return timestamp_series

    def create_ohlc_chart(self, df, symbol, timeframe, indicators, params, highlighted_timestamps=None, chart_position=None):
        """Create OHLC candlestick chart - EXACT copy of original functionality"""
        if df is None or df.empty:
            return None
        
        # Convert UTC timestamps to local time for display
        df_display = df.copy()
        df_display['timestamp_local'] = self.convert_utc_to_local(df['timestamp'])
        
        # Compute Half Trend if selected
        if 'Half Trend' in indicators:
            df_display = self.calculate_half_trend(df_display, period=params['Half Trend']['period'],
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
                x=df_display['timestamp_local'],
                open=df_display['open'],
                high=df_display['high'],
                low=df_display['low'],
                close=df_display['close'],
                name='Price',
                increasing_line_color=BULLISH_COLOR,
                decreasing_line_color=BEARISH_COLOR,
                increasing_fillcolor=BULLISH_COLOR,
                decreasing_fillcolor=BEARISH_COLOR
            ),
            row=1, col=1
        )
        
        # Add highlighting overlays using scatter markers based on timestamps - EXACT original logic
        if highlighted_timestamps:
            price_range = df_display['high'].max() - df_display['low'].min()
            
            # Check if DataFrame timestamps are timezone-aware or naive
            df_tz_aware = hasattr(df['timestamp'].iloc[0], 'tz') and df['timestamp'].iloc[0].tz is not None
            
            for timestamp in highlighted_timestamps:
                # Convert timestamp to match DataFrame format
                if hasattr(timestamp, 'tz_localize'):
                    # Already a pandas timestamp
                    if df_tz_aware:
                        # DataFrame is timezone-aware, ensure timestamp is UTC
                        if timestamp.tz is None:
                            timestamp_pd = timestamp.tz_localize('UTC')
                        else:
                            timestamp_pd = timestamp.tz_convert('UTC')
                    else:
                        # DataFrame is timezone-naive, make timestamp naive
                        if timestamp.tz is not None:
                            timestamp_pd = timestamp.tz_convert('UTC').tz_localize(None)
                        else:
                            timestamp_pd = timestamp
                else:
                    # Convert datetime to pandas timestamp
                    if df_tz_aware:
                        # DataFrame is timezone-aware, convert to UTC pandas timestamp
                        if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                            timestamp_pd = pd.Timestamp(timestamp).tz_convert('UTC')
                        else:
                            timestamp_pd = pd.Timestamp(timestamp).tz_localize('UTC')
                    else:
                        # DataFrame is timezone-naive, convert to naive pandas timestamp
                        if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                            timestamp_pd = pd.Timestamp(timestamp).tz_convert('UTC').tz_localize(None)
                        else:
                            timestamp_pd = pd.Timestamp(timestamp)
                
                # Find the row with matching timestamp
                matching_rows = df[df['timestamp'] == timestamp_pd]
                
                # If exact match not found, find the closest earlier timestamp
                if matching_rows.empty:
                    # Find all timestamps that are earlier than or equal to the target
                    earlier_timestamps = df[df['timestamp'] <= timestamp_pd]['timestamp']
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
                    candle_index = matching_rows.index[0]
                    # Get the local timestamp for display
                    local_timestamp = df_display.loc[candle_index, 'timestamp_local']
                    
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
                    
                    # Format timestamps for display (convert to local time for display)
                    if hasattr(timestamp, 'strftime'):
                        # Convert to local time for display
                        try:
                            import time
                            is_dst = time.daylight and time.localtime().tm_isdst > 0
                            utc_offset_seconds = - (time.altzone if is_dst else time.timezone)
                            local_tz_offset = timedelta(seconds=utc_offset_seconds)
                            
                            if hasattr(timestamp, 'tz') and timestamp.tz is not None:
                                local_timestamp = timestamp.tz_convert(None) + local_tz_offset
                            else:
                                local_timestamp = timestamp + local_tz_offset
                            target_str = local_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            target_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        # Convert UTC pandas timestamp to local time for display
                        try:
                            import time
                            is_dst = time.daylight and time.localtime().tm_isdst > 0
                            utc_offset_seconds = - (time.altzone if is_dst else time.timezone)
                            local_tz_offset = timedelta(seconds=utc_offset_seconds)
                            local_timestamp = timestamp_pd + local_tz_offset
                            target_str = local_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            target_str = timestamp_pd.strftime("%Y-%m-%d %H:%M:%S")
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[local_timestamp],
                            y=[star_y],
                            mode='markers',
                            marker=dict(
                                symbol='star',
                                size=20,
                                color=star_color,
                                line=dict(width=2, color=border_color)
                            ),
                            name=f'Highlight {target_str.split()[1][:8]}',
                            showlegend=False,
                            hovertext=f'Highlighted Candlestick<br>Target: {target_str}<br>Actual: {local_timestamp.strftime("%Y-%m-%d %H:%M:%S")}<br>High: ${candle["high"]:.4f}<br>Close: ${candle["close"]:.4f}{hover_suffix}'
                        ),
                        row=1, col=1
                    )
        
        fig.update_yaxes(title_text='Price (USDT)', row=1, col=1)

        # Plot Half Trend
        if 'Half Trend' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=df_display['timestamp_local'], 
                    y=df_display['half_trend'],
                    mode='lines', 
                    name='Half Trend',
                    line=dict(width=1)
                ), row=1, col=1
            )

        # Bollinger Band
        if "Bollinger Band" in indicators:
            w = params['Bollinger Band']['window']
            dev = params['Bollinger Band']['window_dev']
            bb = ta.volatility.BollingerBands(close=df_display["close"], window=w, window_dev=dev)

            df_display["bb_middle"] = bb.bollinger_mavg()
            df_display["bb_upper"]  = bb.bollinger_hband()
            df_display["bb_lower"]  = bb.bollinger_lband()

            fig.add_trace(
                go.Scatter(
                    x=df_display["timestamp_local"],
                    y=df_display["bb_middle"],
                    mode="lines",
                    line=dict(color="rgba(255,0,0,1)", width=1),
                    opacity = 0.6,
                    name="BB Middle (20)"
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df_display["timestamp_local"],
                    y=df_display["bb_upper"],
                    mode="lines",
                    line=dict(color="rgba(51,153,255,0.8)", width=1),
                    name="BB Upper (20,+2σ)",
                    hovertemplate="Upper: %{y:.2f}<extra></extra>"
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df_display["timestamp_local"],
                    y=df_display["bb_lower"],
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
                df_display["rsi"] = (ta.momentum.RSIIndicator(df_display["close"], window=w).rsi())

                fig.add_trace(
                    go.Scatter(
                        x=df_display['timestamp_local'],
                        y=df_display['rsi'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='white', width=1)
                    ),
                    row=idx, col=1
                )
                
                # Add horizontal lines for overbought and oversold levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=idx, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=idx, col=1)
                
                fig.update_yaxes(title_text='RSI', row=idx, col=1)

            # Williams %R Chart
            if ind == "William % Range":
                lbp = params['William % Range']['lbp']
                df_display["williams_r"] = ta.momentum.williams_r(df_display["high"], df_display["low"], df_display["close"], lbp)

                fig.add_trace(
                    go.Scatter(
                        x=df_display['timestamp_local'],
                        y=df_display['williams_r'],
                        mode='lines',
                        name='Williams %R',
                        line=dict(color='orange', width=1)
                    ),
                    row=idx, col=1
                )
                
                # Add horizontal lines for overbought and oversold levels
                fig.add_hline(y=-20, line_dash="dash", line_color="red", row=idx, col=1)
                fig.add_hline(y=-80, line_dash="dash", line_color="green", row=idx, col=1)
                
                fig.update_yaxes(title_text='Williams %R', row=idx, col=1)

            # KDJ Chart
            if ind == "KDJ":
                period = params['KDJ']['period']
                signal = params['KDJ']['signal']
                
                # Calculate KDJ
                lowest_low = df_display['low'].rolling(window=period).min()
                highest_high = df_display['high'].rolling(window=period).max()
                
                k_percent = 100 * (df_display['close'] - lowest_low) / (highest_high - lowest_low)
                k_percent = k_percent.rolling(window=signal).mean()
                d_percent = k_percent.rolling(window=signal).mean()
                j_percent = 3 * k_percent - 2 * d_percent
                
                df_display['k_percent'] = k_percent
                df_display['d_percent'] = d_percent
                df_display['j_percent'] = j_percent

                fig.add_trace(
                    go.Scatter(
                        x=df_display['timestamp_local'],
                        y=df_display['k_percent'],
                        mode='lines',
                        name='K%',
                        line=dict(color='blue', width=1)
                    ),
                    row=idx, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_display['timestamp_local'],
                        y=df_display['d_percent'],
                        mode='lines',
                        name='D%',
                        line=dict(color='red', width=1)
                    ),
                    row=idx, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_display['timestamp_local'],
                        y=df_display['j_percent'],
                        mode='lines',
                        name='J%',
                        line=dict(color='green', width=1)
                    ),
                    row=idx, col=1
                )
                
                # Add horizontal lines for overbought and oversold levels
                fig.add_hline(y=80, line_dash="dash", line_color="red", row=idx, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color="green", row=idx, col=1)
                
                fig.update_yaxes(title_text='KDJ', row=idx, col=1)
        
        # Update layout - EXACT original styling
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
        
        # Auto-pan chart based on navigation position or highlighted timestamps
        if chart_position is not None:
            # Use navigation position (chart_position is an index into the dataframe)
            if 0 <= chart_position < len(df):
                end_time = df_display.iloc[chart_position]['timestamp_local']
                
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
                data_start = df_display['timestamp_local'].min()
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
            
            # Check if DataFrame timestamps are timezone-aware or naive
            df_tz_aware = hasattr(df['timestamp'].iloc[0], 'tz') and df['timestamp'].iloc[0].tz is not None
            
            # Convert highlight timestamp to match DataFrame format
            if hasattr(latest_highlight, 'tz_localize'):
                # Already a pandas timestamp
                if df_tz_aware:
                    # DataFrame is timezone-aware, ensure highlight is UTC
                    if latest_highlight.tz is None:
                        latest_highlight_pd = latest_highlight.tz_localize('UTC')
                    else:
                        latest_highlight_pd = latest_highlight.tz_convert('UTC')
                else:
                    # DataFrame is timezone-naive, make highlight naive
                    if latest_highlight.tz is not None:
                        latest_highlight_pd = latest_highlight.tz_convert('UTC').tz_localize(None)
                    else:
                        latest_highlight_pd = latest_highlight
            else:
                # Convert datetime to pandas timestamp
                if df_tz_aware:
                    # DataFrame is timezone-aware, convert to UTC pandas timestamp
                    if hasattr(latest_highlight, 'tzinfo') and latest_highlight.tzinfo is not None:
                        latest_highlight_pd = pd.Timestamp(latest_highlight).tz_convert('UTC')
                    else:
                        latest_highlight_pd = pd.Timestamp(latest_highlight).tz_localize('UTC')
                else:
                    # DataFrame is timezone-naive, convert to naive pandas timestamp
                    if hasattr(latest_highlight, 'tzinfo') and latest_highlight.tzinfo is not None:
                        latest_highlight_pd = pd.Timestamp(latest_highlight).tz_convert('UTC').tz_localize(None)
                    else:
                        latest_highlight_pd = pd.Timestamp(latest_highlight)
            
            # Find the actual timestamp to use (exact match or closest earlier)
            matching_rows = df[df['timestamp'] == latest_highlight_pd]
            if matching_rows.empty:
                earlier_timestamps = df[df['timestamp'] <= latest_highlight_pd]['timestamp']
                if not earlier_timestamps.empty:
                    actual_timestamp = earlier_timestamps.max()
                    matching_rows = df[df['timestamp'] == actual_timestamp]
                else:
                    actual_timestamp = latest_highlight_pd
                    matching_rows = df.iloc[-1:]  # Use last row as fallback
            else:
                actual_timestamp = latest_highlight_pd
            
            # Get the corresponding local timestamp for display
            if not matching_rows.empty:
                candle_index = matching_rows.index[0]
                actual_local_timestamp = df_display.loc[candle_index, 'timestamp_local']
            else:
                # Fallback to converting UTC to local
                actual_local_timestamp = self.convert_utc_to_local(pd.Series([actual_timestamp])).iloc[0]
            
            # Calculate time window to show around the highlighted timestamp
            timeframe_minutes = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720, '1d': 1440
            }
            
            tf_minutes = timeframe_minutes.get(timeframe, 60)
            time_window_minutes = tf_minutes * 30  # Show 30 candles before and after
            
            start_time = actual_local_timestamp - pd.Timedelta(minutes=time_window_minutes)
            end_time = actual_local_timestamp + pd.Timedelta(minutes=time_window_minutes)
            
            # Ensure we don't go outside our data range
            data_start = df_display['timestamp_local'].min()
            data_end = df_display['timestamp_local'].max()
            if start_time < data_start:
                start_time = data_start
            if end_time > data_end:
                end_time = data_end
            
            fig.update_layout(
                xaxis=dict(
                    range=[start_time, end_time],
                    type='date'
                )
            )
        
        return fig

    def _handle_highlighting_navigation(self, crypto, highlighted_timestamps, df, current_position, timeframe_changed, nav_key):
        """Handle auto-panning to highlighted candlesticks"""
        
        # Check for forced navigation (when user clicks "Go To" button)
        force_nav_key = f"force_nav_{crypto}"
        if force_nav_key in st.session_state:
            forced_timestamp = st.session_state[force_nav_key]
            del st.session_state[force_nav_key]  # Clear the forced navigation
            
            # Find the best match for forced navigation
            best_match_position = None
            best_time_diff = None
            
            # Convert forced timestamp to a comparable format
            if hasattr(forced_timestamp, 'tz') and forced_timestamp.tz is not None:
                search_timestamp = forced_timestamp.tz_convert('UTC').tz_localize(None)
            elif hasattr(forced_timestamp, 'tzinfo') and forced_timestamp.tzinfo is not None:
                # Handle datetime objects with tzinfo
                search_timestamp = pd.Timestamp(forced_timestamp).tz_convert('UTC').tz_localize(None)
            else:
                search_timestamp = pd.Timestamp(forced_timestamp)
            
            # Search through all timestamps in the dataframe
            for idx, row in df.iterrows():
                df_timestamp = row['timestamp']
                
                # Convert df timestamp to comparable format
                if hasattr(df_timestamp, 'tz') and df_timestamp.tz is not None:
                    compare_timestamp = df_timestamp.tz_convert('UTC').tz_localize(None)
                elif hasattr(df_timestamp, 'tzinfo') and df_timestamp.tzinfo is not None:
                    # Handle datetime objects with tzinfo
                    compare_timestamp = pd.Timestamp(df_timestamp).tz_convert('UTC').tz_localize(None)
                else:
                    compare_timestamp = pd.Timestamp(df_timestamp)
                
                # Calculate time difference - ensure both are pandas Timestamps
                try:
                    time_diff = abs((search_timestamp - compare_timestamp).total_seconds())
                except Exception:
                    # Fallback: convert both to naive datetime
                    search_dt = search_timestamp.to_pydatetime() if hasattr(search_timestamp, 'to_pydatetime') else search_timestamp
                    compare_dt = compare_timestamp.to_pydatetime() if hasattr(compare_timestamp, 'to_pydatetime') else compare_timestamp
                    
                    # Remove timezone info if present
                    if hasattr(search_dt, 'tzinfo') and search_dt.tzinfo is not None:
                        search_dt = search_dt.replace(tzinfo=None)
                    if hasattr(compare_dt, 'tzinfo') and compare_dt.tzinfo is not None:
                        compare_dt = compare_dt.replace(tzinfo=None)
                    
                    time_diff = abs((search_dt - compare_dt).total_seconds())
                
                # Keep track of the best match
                if best_time_diff is None or time_diff < best_time_diff:
                    best_time_diff = time_diff
                    best_match_position = idx
            
            # Force navigation to the selected highlight
            if best_match_position is not None:
                st.session_state[nav_key] = best_match_position
                current_position = best_match_position
                return current_position
        
        if highlighted_timestamps and not timeframe_changed:
            # Check if this is a new highlight
            prev_highlights_key = f"prev_highlighted_candles_{crypto}"
            prev_highlights = st.session_state.get(prev_highlights_key, [])
            
            # If we have new highlights, auto-pan to the latest one
            if len(highlighted_timestamps) > len(prev_highlights):
                latest_highlight = max(highlighted_timestamps)
                
                # Simplified approach: find the closest matching timestamp
                best_match_position = None
                best_time_diff = None
                
                # Convert latest_highlight to a comparable format
                if hasattr(latest_highlight, 'tz') and latest_highlight.tz is not None:
                    # If timezone-aware, convert to UTC for comparison
                    search_timestamp = latest_highlight.tz_convert('UTC').tz_localize(None)
                elif hasattr(latest_highlight, 'tzinfo') and latest_highlight.tzinfo is not None:
                    # Handle datetime objects with tzinfo
                    search_timestamp = pd.Timestamp(latest_highlight).tz_convert('UTC').tz_localize(None)
                else:
                    # If timezone-naive, use as-is
                    search_timestamp = pd.Timestamp(latest_highlight)
                
                # Search through all timestamps in the dataframe
                for idx, row in df.iterrows():
                    df_timestamp = row['timestamp']
                    
                    # Convert df timestamp to comparable format
                    if hasattr(df_timestamp, 'tz') and df_timestamp.tz is not None:
                        compare_timestamp = df_timestamp.tz_convert('UTC').tz_localize(None)
                    elif hasattr(df_timestamp, 'tzinfo') and df_timestamp.tzinfo is not None:
                        # Handle datetime objects with tzinfo
                        compare_timestamp = pd.Timestamp(df_timestamp).tz_convert('UTC').tz_localize(None)
                    else:
                        compare_timestamp = pd.Timestamp(df_timestamp)
                    
                    # Calculate time difference - ensure both are pandas Timestamps
                    try:
                        time_diff = abs((search_timestamp - compare_timestamp).total_seconds())
                    except Exception:
                        # Fallback: convert both to naive datetime
                        search_dt = search_timestamp.to_pydatetime() if hasattr(search_timestamp, 'to_pydatetime') else search_timestamp
                        compare_dt = compare_timestamp.to_pydatetime() if hasattr(compare_timestamp, 'to_pydatetime') else compare_timestamp
                        
                        # Remove timezone info if present
                        if hasattr(search_dt, 'tzinfo') and search_dt.tzinfo is not None:
                            search_dt = search_dt.replace(tzinfo=None)
                        if hasattr(compare_dt, 'tzinfo') and compare_dt.tzinfo is not None:
                            compare_dt = compare_dt.replace(tzinfo=None)
                        
                        time_diff = abs((search_dt - compare_dt).total_seconds())
                    
                    # Keep track of the best match
                    if best_time_diff is None or time_diff < best_time_diff:
                        best_time_diff = time_diff
                        best_match_position = idx
                
                # If we found a good match (within 1 hour), update navigation
                if best_match_position is not None and best_time_diff <= 3600:  # 1 hour tolerance
                    st.session_state[nav_key] = best_match_position
                    current_position = best_match_position
                
                # Update the previous highlights state
                st.session_state[prev_highlights_key] = highlighted_timestamps.copy()
        
        elif highlighted_timestamps and timeframe_changed:
            # When timeframe changes and we have highlights, find the highlighted position in new timeframe
            latest_highlight = max(highlighted_timestamps)
            
            # Simplified approach for timeframe changes
            best_match_position = None
            best_time_diff = None
            
            # Convert latest_highlight to a comparable format
            if hasattr(latest_highlight, 'tz') and latest_highlight.tz is not None:
                search_timestamp = latest_highlight.tz_convert('UTC').tz_localize(None)
            elif hasattr(latest_highlight, 'tzinfo') and latest_highlight.tzinfo is not None:
                # Handle datetime objects with tzinfo
                search_timestamp = pd.Timestamp(latest_highlight).tz_convert('UTC').tz_localize(None)
            else:
                search_timestamp = pd.Timestamp(latest_highlight)
            
            # Search through all timestamps in the new timeframe dataframe
            for idx, row in df.iterrows():
                df_timestamp = row['timestamp']
                
                # Convert df timestamp to comparable format
                if hasattr(df_timestamp, 'tz') and df_timestamp.tz is not None:
                    compare_timestamp = df_timestamp.tz_convert('UTC').tz_localize(None)
                elif hasattr(df_timestamp, 'tzinfo') and df_timestamp.tzinfo is not None:
                    # Handle datetime objects with tzinfo
                    compare_timestamp = pd.Timestamp(df_timestamp).tz_convert('UTC').tz_localize(None)
                else:
                    compare_timestamp = pd.Timestamp(df_timestamp)
                
                # Calculate time difference - ensure both are pandas Timestamps
                try:
                    time_diff = abs((search_timestamp - compare_timestamp).total_seconds())
                except Exception:
                    # Fallback: convert both to naive datetime
                    search_dt = search_timestamp.to_pydatetime() if hasattr(search_timestamp, 'to_pydatetime') else search_timestamp
                    compare_dt = compare_timestamp.to_pydatetime() if hasattr(compare_timestamp, 'to_pydatetime') else compare_timestamp
                    
                    # Remove timezone info if present
                    if hasattr(search_dt, 'tzinfo') and search_dt.tzinfo is not None:
                        search_dt = search_dt.replace(tzinfo=None)
                    if hasattr(compare_dt, 'tzinfo') and compare_dt.tzinfo is not None:
                        compare_dt = compare_dt.replace(tzinfo=None)
                    
                    time_diff = abs((search_dt - compare_dt).total_seconds())
                
                # Keep track of the best match
                if best_time_diff is None or time_diff < best_time_diff:
                    best_time_diff = time_diff
                    best_match_position = idx
            
            # Update navigation to the best match in new timeframe
            if best_match_position is not None:
                st.session_state[nav_key] = best_match_position
                current_position = best_match_position
        
        return current_position

    def _render_highlight_controls(self, crypto, idx, df, highlight_key):
        """Render highlighting controls - EXACT original functionality"""
        st.subheader("🎯 Highlight Candlesticks")
        
        col1, col2 = st.columns(2)
        with col1:
            # Date picker for highlighting
            highlight_date = st.date_input(
                "Select Date to Highlight",
                value=datetime.now().date(),
                key=f"highlight_date_{crypto}_{idx}"
            )
        
        with col2:
            # Time picker for highlighting - adjust based on timeframe
            if self.timeframe == '5m':
                # For 5-minute timeframe, use selectbox with 5-minute intervals
                # Generate all possible 5-minute intervals in a day
                time_options = []
                for hour in range(24):
                    for minute in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
                        time_str = f"{hour:02d}:{minute:02d}"
                        time_options.append(time_str)
                
                # Default to current time rounded to nearest 5-minute interval
                current_time = datetime.now().time()
                rounded_minutes = (current_time.minute // 5) * 5
                default_time_str = f"{current_time.hour:02d}:{rounded_minutes:02d}"
                
                # Find the index of default time
                try:
                    default_index = time_options.index(default_time_str)
                except ValueError:
                    default_index = 0
                
                selected_time_str = st.selectbox(
                    "Select Time to Highlight (Local)",
                    options=time_options,
                    index=default_index,
                    key=f"highlight_time_{crypto}_{idx}",
                    help="Time in 5-minute intervals (e.g., 14:00, 14:05, 14:10)"
                )
                
                # Parse the selected time string back to time object
                hour, minute = map(int, selected_time_str.split(':'))
                highlight_time = datetime.now().time().replace(hour=hour, minute=minute, second=0, microsecond=0)
                
            else:
                # For other timeframes, use regular time input
                default_time = datetime.now().time().replace(second=0, microsecond=0)
                highlight_time = st.time_input(
                    "Select Time to Highlight (Local)",
                    value=default_time,
                    key=f"highlight_time_{crypto}_{idx}",
                    help="Select time in local timezone"
                )
        
        # Combine date and time, make timezone-aware
        highlight_datetime = datetime.combine(highlight_date, highlight_time)
        # Convert local time to UTC for comparison with chart data
        try:
            # Get the system's local timezone offset
            import time
            is_dst = time.daylight and time.localtime().tm_isdst > 0
            utc_offset_seconds = - (time.altzone if is_dst else time.timezone)
            
            # Create timezone-aware datetime in local timezone
            local_tz_offset = timedelta(seconds=utc_offset_seconds)
            # Manually convert to UTC
            highlight_datetime_utc = highlight_datetime - local_tz_offset
            highlight_datetime = pytz.UTC.localize(highlight_datetime_utc)
        except:
            # Fallback: treat as UTC if timezone conversion fails
            highlight_datetime = pytz.UTC.localize(highlight_datetime)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➕ Add Highlight", key=f"add_highlight_{crypto}_{idx}"):
                current_highlights = st.session_state.get(highlight_key, [])
                
                # Check if this timestamp already exists
                if highlight_datetime not in current_highlights:
                    current_highlights.append(highlight_datetime)
                    st.session_state[highlight_key] = current_highlights
                    # Show the time in local timezone for user confirmation
                    try:
                        import time
                        is_dst = time.daylight and time.localtime().tm_isdst > 0
                        utc_offset_seconds = - (time.altzone if is_dst else time.timezone)
                        local_tz_offset = timedelta(seconds=utc_offset_seconds)
                        local_time = (highlight_datetime.replace(tzinfo=None) + local_tz_offset)
                        display_time = local_time.strftime('%Y-%m-%d %H:%M Local')
                    except:
                        display_time = highlight_datetime.strftime('%Y-%m-%d %H:%M UTC')
                    st.success(f"Added highlight for {display_time}")
                    st.rerun()
                else:
                    st.warning("This timestamp is already highlighted!")
        
        with col2:
            if st.button("🗑️ Clear All", key=f"clear_highlights_{crypto}_{idx}"):
                st.session_state[highlight_key] = []
                # Also clear the previous highlights state for auto-pan
                prev_highlights_key = f"prev_highlighted_candles_{crypto}"
                if prev_highlights_key in st.session_state:
                    del st.session_state[prev_highlights_key]
                st.success("Cleared all highlights")
                st.rerun()
        
        with col3:
            # Quick highlight current candlestick
            nav_key = f"chart_nav_{crypto}"
            current_position = st.session_state.get(nav_key, len(df) - 1)
            if st.button("⭐ Highlight Current", key=f"highlight_current_{crypto}_{idx}"):
                if current_position < len(df):
                    current_timestamp = df.iloc[current_position]['timestamp']
                    current_highlights = st.session_state.get(highlight_key, [])
                    
                    if current_timestamp not in current_highlights:
                        current_highlights.append(current_timestamp)
                        st.session_state[highlight_key] = current_highlights
                        st.success(f"Highlighted candlestick at position {current_position + 1}")
                        st.rerun()
                    else:
                        st.warning("Current candlestick is already highlighted!")
