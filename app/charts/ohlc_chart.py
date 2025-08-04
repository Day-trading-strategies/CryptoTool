import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
from datetime import datetime

from app.config import *

from app.indicators.half_trend import HalfTrendIndicator
from app.indicators.bollinger_bands import BollingerBandsIndicator
from app.indicators.rsi import RSIIndicator
from app.indicators.williams_r import WilliamsRIndicator
from app.indicators.kdj import KDJIndicator
from app.charts.chart_navigation import ChartNavigation
from app.charts.candlestick_highlighter import CandlestickHighlighter

class OHLCChartCreator:
    """Class to create OHLC charts with optional indicators."""
    
    def __init__(self, selected_cryptos, timeframe, data_fetcher, selected_indicators, indicator_params, states, highlighted_timestamps=None, chart_position=None):
        self.df = {crypto: [] for crypto in selected_cryptos}
        self.selected_cryptos = selected_cryptos
        self.selected_indicators = selected_indicators if selected_indicators is not None else []
        self.indicator_params = indicator_params
        self.timeframe = timeframe
        self.data_fetcher = data_fetcher
        self.states = states
        self.highlighted_timestamps = highlighted_timestamps
        self.chart_position = chart_position
        self.render()

    def render(self):
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

        st.subheader("📈 OHLC Charts")
        if self.states.bt_mode:
            print("bt_mode on")
            crypto = self.states.crypto

            # if timeframe is {something} 
            # self.states.df is {csvs_name}.csv
            if self.timeframe == "15m":
                self.states.df = pd.read_csv("data/15m_df.csv", parse_dates=["timestamp"])
            if self.timeframe == "5m":
                self.states.df = pd.read_csv("data/5m_df.csv", parse_dates=["timestamp"])
            if self.timeframe == "3m":
                self.states.df = pd.read_csv("data/3m_df.csv", parse_dates=["timestamp"])
            if self.timeframe == "1m":
                self.states.df = pd.read_csv("data/1m_df.csv", parse_dates=["timestamp"])
            if self.timeframe == "1h":
                self.states.df = pd.read_csv("data/1h_df.csv", parse_dates=["timestamp"])
            if self.timeframe =="4h":
                self.states.df = pd.read_csv("data/4h_df.csv", parse_dates=["timestamp"])
            if self.timeframe == "1d":
                self.states.df = pd.read_csv("data/1d.csv", parse_dates=["timestamp"])

            self.df[crypto] = self.states.df
            # Create component instances for backtest
            navigator = ChartNavigation(crypto, self.df[crypto], self.states)
            highlighter = CandlestickHighlighter(crypto, self.df[crypto], self.states)
        
            # Render navigation and highlighting controls
            highlighted_timestamps = highlighter.render()
            current_position = navigator.render()


            fig = self.create_chart(crypto, highlighted_timestamps, current_position)
            self.df[crypto].to_csv("data/backtest_data.csv", index=False)

            # if user chooses a hit point to test, create chart from start to hit point
            if self.states.chart_end:
                temp_df = self.df[crypto].copy()
                end_idx = self.states.chart_navigation[crypto]
                self.df[crypto] = self.df[crypto].iloc[: end_idx + 1]
                
                # Adjust current_position for truncated data
                adjusted_position = current_position if current_position is not None and current_position <= end_idx else None
                
                fig = self.create_chart(crypto, highlighted_timestamps, adjusted_position)
                self.df[crypto] = temp_df
            else:
                # create chart with original data (entire dataset)
                current_position = navigator.navigate_to_latest()
                print(current_position)

                fig = self.create_chart(crypto, highlighted_timestamps, current_position)

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

                hits = pd.read_csv("data/filtered_backtest.csv")
                timestamps = hits["timestamp"].tolist()
                
                if self.states.chart_end in timestamps:
                    self.states.hit_index = timestamps.index(self.states.chart_end)
                else:
                    self.states.hit_index = 0

                # render a “Next →” button
                if timestamps and st.button("Next →", key="next_hit"):
                    current_idx = self.states.hit_index
                    next_idx = (current_idx + 1) % len(timestamps)
                    self.states.hit_index = next_idx

                    # move chart_end to that next timestamp
                    next_ts = timestamps[next_idx]
                    self.states.chart_end = next_ts

                    # find & store its row index
                    chart_end_index = int(
                        self.df[crypto][self.df[crypto]['timestamp'] == next_ts]
                            .index[0]
                    )
                    self.states.chart_navigation[crypto] = chart_end_index

                    # Keep highlighted list in sync with navigation
                    if crypto not in self.states.highlighted_candles:
                        self.states.highlighted_candles[crypto] = []
                    
                    # Add the new timestamp to highlights if not already present
                    if next_ts not in self.states.highlighted_candles[crypto]:
                        self.states.highlighted_candles[crypto].append(next_ts)

                    st.rerun()
                        
                try:
                    filtered = pd.read_csv("data/filtered_backtest.csv")
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
                                # store the clicked timestamp
                                self.states.chart_end = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")

                                # now do a safe ≤ comparison
                                ts_series   = self.df[crypto]["timestamp"]
                                desired_end = (
                                    self.states.chart_end
                                    if self.states.chart_end is not None
                                    else ts_series.max()
                                )
                                valid_ts = ts_series[ts_series <= desired_end]

                                if not valid_ts.empty:
                                    nearest_ts      = valid_ts.max()
                                    chart_end_index = int(ts_series[ts_series == nearest_ts].index[0])
                                else:
                                    chart_end_index = len(ts_series) - 1

                                self.states.chart_navigation[crypto] = chart_end_index
                                st.rerun()
                                # # parse back into a datetime
                                # self.states.chart_end = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                                # chart_end_index = int(self.df[crypto][self.df[crypto]['timestamp'] == self.states.chart_end].index[0])
                                # self.states.chart_navigation[crypto] = chart_end_index
                                # match_index = self.df[crypto][self.df[crypto]['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S") == self.states.chart_end.strftime("%Y-%m-%d %H:%M:%S")].index[0]
                                # st.rerun()

            else:
                st.error(f"Unable to Backtest Chart")
        # Create tabs for each cryptocurrency
        elif len(self.selected_cryptos) > 1:
            tabs = st.tabs(self.selected_cryptos)
            for idx, crypto in enumerate(self.selected_cryptos):
                with tabs[idx]:
                    self.df[crypto] = self.data_fetcher.fetch_ohlc_data(AVAILABLE_CRYPTOS[crypto], self.timeframe)
                    if self.df[crypto] is not None:
                        # Create component instances for this crypto/tab
                        navigator = ChartNavigation(crypto, self.df[crypto], self.states)
                        highlighter = CandlestickHighlighter(crypto, self.df[crypto], self.states)

                        # Set unique keys for tab components
                        navigator.tab_idx = idx
                        highlighter.tab_idx = idx

                        # Render navigation and highlighting controls
                        highlighted_timestamps = highlighter.render()
                        current_position = navigator.render()

                        fig = self.create_chart(crypto, highlighted_timestamps, current_position)
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
                                st.dataframe(self.df[crypto].tail(10).iloc[::-1], use_container_width=True)
                    else:
                        st.error(f"Unable to load chart for {crypto}")
        else:
            crypto = self.selected_cryptos[0]
            self.df[crypto] = self.data_fetcher.fetch_ohlc_data(AVAILABLE_CRYPTOS[crypto], self.timeframe)
            if self.df[crypto] is not None:
                # Create component instances for single crypto
                navigator = ChartNavigation(crypto, self.df[crypto], self.states)
                highlighter = CandlestickHighlighter(crypto, self.df[crypto], self.states)
            
                # Render navigation and highlighting controls
                highlighted_timestamps = highlighter.render()
                current_position = navigator.render()
            
                # CREATE CHART
                fig = self.create_chart(crypto, highlighted_timestamps, current_position)
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
                        st.dataframe(self.df[crypto].tail(10).iloc[::-1], use_container_width=True)
            else:
                st.error(f"Unable to load chart for {crypto}")

    def _add_highlight_markers(self, fig, df, highlighted_timestamps):
        """Add star markers for highlighted timestamps"""
        price_range = df['high'].max() - df['low'].min()
        
        for timestamp in highlighted_timestamps:
            # Find matching row (exact or closest earlier)
            matching_rows = df[df['timestamp'] == timestamp]
            
            if matching_rows.empty:
                earlier_timestamps = df[df['timestamp'] <= timestamp]['timestamp']
                if not earlier_timestamps.empty:
                    closest_timestamp = earlier_timestamps.max()
                    matching_rows = df[df['timestamp'] == closest_timestamp]
                    is_approximate = True
                else:
                    continue
            else:
                is_approximate = False
            
            if not matching_rows.empty:
                candle = matching_rows.iloc[0]
                star_y = candle['high'] + (price_range * 0.05)
                
                star_color = '#FFA500' if is_approximate else '#FFD700'
                border_color = '#FF6347' if is_approximate else '#FF8C00'
                hover_suffix = '<br><i>Approximate match (closest earlier)</i>' if is_approximate else ''
                
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

    def _add_auto_panning(self, fig, df, chart_position, highlighted_timestamps, timeframe, crypto, start_time=None):
        """Add auto-panning functionality to focus on navigation or highlights"""
        
        if getattr(self.states, "chart_end", None) is not None:
            end_time = self.states.chart_end               # right-most x-axis value

            # Optional: if you’d rather snap to an actual candle when one exists,
            # uncomment the 5 lines below.
            # ts_series = df['timestamp']
            # valid_ts  = ts_series[ts_series <= end_time]
            # if not valid_ts.empty:
            #     end_time = valid_ts.max()                # nearest ≤ highlight

            start_time = self._calculate_window_start(end_time, timeframe, df)
            fig.update_xaxes(range=[start_time, end_time], type='date')

            # Keep navigation pointer consistent so Back/Forward buttons work
            ts_series = df['timestamp']
            nav_idx   = (
                int(ts_series[ts_series == end_time].index[0])
                if end_time in set(ts_series)
                else int(ts_series[ts_series <= end_time].index.max())
                if not ts_series[ts_series <= end_time].empty
                else len(df) - 1
            )
            nav_positions              = self.states.chart_navigation
            nav_positions[crypto]      = nav_idx
            self.states.chart_navigation = nav_positions
            return  # highlight takes absolute priority; skip other auto-pan logic

        # Check if we have highlights first
        if highlighted_timestamps:
            prev_highlights_key = f"prev_highlighted_candles_{crypto}"
            prev_highlights = st.session_state.get(prev_highlights_key, [])
            
            # If we have new highlights, use highlight positioning (priority over navigation)
            if len(highlighted_timestamps) > len(prev_highlights):
                latest_highlight = max(highlighted_timestamps)
                
                # Find the index of the highlighted timestamp in the dataframe
                matching_rows = df[df['timestamp'] == latest_highlight]
                if matching_rows.empty:
                    # Find closest earlier timestamp
                    earlier_timestamps = df[df['timestamp'] <= latest_highlight]['timestamp']
                    if not earlier_timestamps.empty:
                        closest_timestamp = earlier_timestamps.max()
                        matching_rows = df[df['timestamp'] == closest_timestamp]
                
                if not matching_rows.empty:
                    # Get the index of the highlighted candle
                    highlight_index = matching_rows.index[0]
                    # Use the same positioning logic as navigation
                    # highlight_time = df.iloc[highlight_index]['timestamp']
                    highlight_time = latest_highlight
                    start_time = self._calculate_window_start(highlight_time, timeframe, df)
                    
                    fig.update_xaxes(
                        range=[start_time, highlight_time],
                        type='date'
                    )
                    
                    # IMPORTANT: Update navigation position to match the highlighted position
                    # This ensures navigation buttons work correctly after highlighting
                    nav_positions = self.states.chart_navigation
                    nav_positions[crypto] = int(highlight_index)  # Ensure it's a Python int, not numpy type
                    self.states.chart_navigation = nav_positions
                    
                    # Update the previous highlights state to prevent re-panning
                    st.session_state[prev_highlights_key] = highlighted_timestamps.copy()
                    return  # Exit early after handling highlight positioning
            
            # If we have existing highlights, check if navigation is different from highlight position
            # If so, navigation buttons are being used and should take priority
            else:
                latest_highlight = max(highlighted_timestamps)
                
                # Find the index of the highlighted timestamp in the dataframe
                matching_rows = df[df['timestamp'] == latest_highlight]
                if matching_rows.empty:
                    # Find closest earlier timestamp for approximate match
                    earlier_timestamps = df[df['timestamp'] <= latest_highlight]['timestamp']
                    if not earlier_timestamps.empty:
                        closest_timestamp = earlier_timestamps.max()
                        matching_rows = df[df['timestamp'] == closest_timestamp]
                
                if not matching_rows.empty:
                    highlight_index = matching_rows.index[0]
                    
                    # Check if navigation position is different from highlight position
                    # If different, user is navigating and navigation should take priority
                    if chart_position is not None and chart_position != highlight_index:
                        # Use navigation position instead of highlight
                        if 0 <= chart_position < len(df):
                            end_time = df.iloc[chart_position]['timestamp']
                            start_time = self._calculate_window_start(end_time, timeframe, df)
                            fig.update_xaxes(range=[start_time, end_time])
                            return
                    
                    # Otherwise, use highlight positioning
                    # highlight_time = df.iloc[highlight_index]['timestamp']
                    highlight_time = latest_highlight
                    start_time = self._calculate_window_start(highlight_time, timeframe, df)
                    
                    fig.update_xaxes(
                        range=[start_time, highlight_time],
                        type='date'
                    )
                    return  # Exit early - highlights take priority when navigation matches
                
                # If no valid highlights found, clear them and fall through to navigation
                else:
                    highlights = self.states.highlighted_candles
                    if crypto in highlights:
                        highlights[crypto] = []
                        self.states.highlighted_candles = highlights
                    # Clear previous highlights state
                    if prev_highlights_key in st.session_state:
                        del st.session_state[prev_highlights_key]
        
        # Use navigation position when no highlights exist or when highlights are cleared
        if chart_position is not None:
            if 0 <= chart_position < len(df):
                end_time = df.iloc[chart_position]['timestamp']
                start_time = self._calculate_window_start(end_time, timeframe, df)
                
                fig.update_xaxes(range=[start_time, end_time])

    def _calculate_window_start(self, end_time, timeframe, df):
        """Calculate the start time for the chart window"""
        timeframe_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720, '1d': 1440
        }
        
        tf_minutes = timeframe_minutes.get(timeframe, 60)
        time_window_minutes = tf_minutes * 150  # Show ~60 candles
        start_time = end_time - pd.Timedelta(minutes=time_window_minutes)
        
        # Ensure start_time is not before data range
        data_start = df['timestamp'].min()
        if start_time < data_start:
            start_time = data_start
        
        return start_time

    def create_chart(self, crypto, highlighted_timestamps=None, chart_position=None) -> go.Figure:
        """Create OHLC candlestick chart with optional highlighting and navigation"""
        if self.df[crypto] is None or self.df[crypto].empty:
            return None
                
        sep_inds = []
        for ind in self.selected_indicators:
            if ind in SEPARATE_AX_INDICATORS:
                sep_inds.append(ind)
        
        # Changes height of chart as more indicators are added.
        n_rows = 1 + len(sep_inds)
        weights = [3] + [1] * len(sep_inds)
        total_weight = sum(weights)
        row_heights = [w/total_weight for w in weights]

        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes = True,
            vertical_spacing=0.02,
            row_heights = row_heights,
            subplot_titles=(f'{crypto} Price Chart ({self.timeframe})',)
        )
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.df[crypto]['timestamp'],
                open=self.df[crypto]['open'],
                high=self.df[crypto]['high'],
                low=self.df[crypto]['low'],
                close=self.df[crypto]['close'],
                name='Price',
                increasing_line_color=BULLISH_COLOR,
                decreasing_line_color=BEARISH_COLOR,
                increasing_fillcolor=BULLISH_COLOR,
                decreasing_fillcolor=BEARISH_COLOR
            ),
            row=1, col=1
        )

        if highlighted_timestamps:
            self._add_highlight_markers(fig, self.df[crypto], highlighted_timestamps)

        fig.update_yaxes(title_text='Price (USDT)', row=1, col=1)
        
        for ind in self.selected_indicators:
            if ind == "RSI":
                indicator = RSIIndicator(**self.indicator_params.get("RSI", {}))
            elif ind == "Bollinger Band":
                indicator = BollingerBandsIndicator(**self.indicator_params.get("Bollinger Band", {}))
            elif ind == "KDJ":
                indicator = KDJIndicator(**self.indicator_params.get("KDJ", {}))
            elif ind == "Half Trend":
                indicator = HalfTrendIndicator(**self.indicator_params.get("Half Trend", {}))
            elif ind == "William % Range":
                indicator = WilliamsRIndicator(**self.indicator_params.get("William % Range", {}))
            else:
                continue

            self.df[crypto] = indicator.calculate(self.df[crypto])

            indicator.add_traces(fig, self.df[crypto], 1 if ind not in sep_inds else sep_inds.index(ind) + 2)
            
        # Update layout
        fig.update_layout(
            title=f'{AVAILABLE_CRYPTOS[crypto]} - {self.timeframe} Chart',
            xaxis_rangeslider_visible=False,
            height=CHART_HEIGHT,
            showlegend=False,
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(color='white'),
            dragmode='pan'  # Set pan as default drag mode in layout
        )
            
        if chart_position is not None or highlighted_timestamps or self.states.chart_end is not None:
            print(f"chart_position is {chart_position}, highlighted_timestamps is {highlighted_timestamps}, backtest is {self.states.bt_mode} and chart_end is {self.states.chart_end}")
            self._add_auto_panning(fig, self.df[crypto], chart_position, highlighted_timestamps, self.timeframe, crypto)
            
        return fig