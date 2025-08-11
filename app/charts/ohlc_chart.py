import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np

from app.config import *

from app.indicators.half_trend import HalfTrendIndicator
from app.indicators.bollinger_bands import BollingerBandsIndicator
from app.indicators.rsi import RSIIndicator
from app.indicators.williams_r import WilliamsRIndicator
from app.indicators.kdj import KDJIndicator
from app.indicators.stochastic import StochasticIndicator
from app.charts.chart_navigation import ChartNavigation
from app.charts.candlestick_highlighter import CandlestickHighlighter
from app.trade_simulator import TradeSimulator

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

            # detects if timeframe has changed, and reassign chart_end to time of last candlestick in the previous timeframe.
            prev_tf = self.states.previous_timeframe
            if prev_tf and prev_tf != self.timeframe:
                old_idx = self.states.chart_navigation.get(crypto)
                old_df  = self.states.df
                if old_idx is not None and old_df is not None and not old_df.empty:
                    old_ts = old_df.iloc[old_idx]["timestamp"]
                    self.states.chart_end = old_ts
                
            # remember that we’ve now initialised this timeframe
            self.states.previous_timeframe = self.timeframe
                    
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

            # detects if we changed timeframes, then reasigns chart_position to the timestamp we converted above.
            if prev_tf and prev_tf != self.timeframe:
                ts_series = self.df[crypto]["timestamp"]
                valid     = ts_series[ts_series <= self.states.chart_end]
                new_idx   = int(valid.index.max()) if not valid.empty else len(ts_series) - 1
                nav_map   = self.states.chart_navigation
                nav_map[crypto] = new_idx
                self.states.chart_navigation = nav_map
                
            # Create component instances for backtest
            navigator = ChartNavigation(crypto, self.df[crypto], self.states)
            highlighter = CandlestickHighlighter(crypto, self.df[crypto], self.states)
        
            # Render navigation and highlighting controls
            highlighted_timestamps = highlighter.render()
            current_position = navigator.render()

            # for hover_indicators
            data_box = st.empty()       # ← this box will update on hover


            fig = self.create_chart(crypto, self.selected_indicators, highlighted_timestamps, current_position)
            self.df[crypto].to_csv("data/backtest_data.csv", index=False)

            # if chart end exists, create chart from start to chart end.
            if self.states.chart_end:
                temp_df = self.df[crypto].copy()
                end_idx = self.states.chart_navigation[crypto]
                self.df[crypto] = self.df[crypto].iloc[: end_idx + 1]
                
                # Adjust current_position for truncated data
                adjusted_position = current_position if current_position is not None and current_position <= end_idx else None
                
                fig = self.create_chart(crypto, self.selected_indicators, highlighted_timestamps, adjusted_position)
                self.df[crypto], temp_df = temp_df, self.df[crypto] 
            else:
                # create chart with original data (entire dataset)
                current_position = navigator.navigate_to_latest()
                print(current_position)

                fig = self.create_chart(crypto, self.selected_indicators, highlighted_timestamps, current_position)

            simulator = TradeSimulator(fig, crypto, temp_df, self.states)
            col1, col2 = st.columns([6,1])

            if fig:
                with col2:
                    simulator.render()

                with col1:
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
                            'modeBarButtonsToAdd': ['drawline', 'eraseshape'],  # enable drawing

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
                

                # Load & clean hits (force datetimes; sort so Next goes chronologically)
                hits = pd.read_csv("data/filtered_backtest.csv", parse_dates=["timestamp"])
                hits = hits.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

                # Optional: keep this if other UI parts still use it
                timestamps = hits["timestamp"].tolist()

                # --- Figure out "now" on the chart (priority: manual nav > chart_end > last candle) ---
                if current_position is not None and 0 <= current_position < len(self.df[crypto]):
                    ref_time = pd.to_datetime(self.df[crypto].iloc[current_position]["timestamp"])
                elif self.states.chart_end is not None:
                    ref_time = pd.to_datetime(self.states.chart_end)
                else:
                    ref_time = pd.to_datetime(self.df[crypto]["timestamp"].iloc[-1])

                # --- Initialize hit_index to the last hit <= ref_time (so Next jumps to the next one) ---
                if not hits.empty:
                    mask = hits["timestamp"] <= ref_time
                    self.states.hit_index = int(hits.index[mask][-1]) if mask.any() else len(hits) - 1
                else:
                    self.states.hit_index = None

                # -----------------------
                # Next → button (unchanged logic, but use 'hits' to get the timestamp)
                # -----------------------
                if not hits.empty and st.button("Next →", key="next_hit"):
                    self.states.hit_index = (self.states.hit_index + 1) % len(hits)
                    next_ts = pd.to_datetime(hits.loc[self.states.hit_index, "timestamp"])
                    self.states.chart_end = next_ts

                    # snap chart to the nearest candle at or before next_ts
                    ts_series = self.df[crypto]["timestamp"]
                    valid_ts  = ts_series[ts_series <= next_ts]
                    if not valid_ts.empty:
                        nearest_ts      = valid_ts.max()
                        chart_end_index = int(ts_series[ts_series == nearest_ts].index[0])
                    else:
                        chart_end_index = len(ts_series) - 1

                    nav_positions         = self.states.chart_navigation
                    nav_positions[crypto] = chart_end_index
                    self.states.chart_navigation = nav_positions

                    st.rerun()

                col1, col2 = st.columns([1,1])
                with col1:
                                            
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

                                    # updating chart_position.
                                    nav = self.states.chart_navigation
                                    nav[crypto] = chart_end_index
                                    self.states.chart_navigation = nav
                                    
                                    st.rerun()
                with col2:
                    trades = self.states.trading_info["history"]
                    with st.expander(f"Trades List"):
                        if not trades.empty:
                            trades["entry_time"] = pd.to_datetime(trades["entry_time"], errors="coerce")
                            trades = trades.dropna(subset=["entry_time"]).copy()

                            # Precompute a display string and ensure numeric result
                            trades["entry_time_str"] = trades["entry_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

                            if trades.empty:
                                st.write("No trades found")
                            else:
                                # Iterate rows so we can grab the matching result easily
                                for idx, row in trades.iterrows():
                                    col_btn, col_val = st.columns([4, 1])

                                    with col_btn:
                                        if st.button(row["entry_time_str"], key=f"trade_btn_{idx}"):
                                            # store the clicked timestamp
                                            self.states.chart_end = row["entry_time"].to_pydatetime()

                                            # now do a safe ≤ comparison
                                            ts_series   = self.df[crypto]["timestamp"]
                                            desired_end = self.states.chart_end if self.states.chart_end is not None else ts_series.max()
                                            valid_ts    = ts_series[ts_series <= desired_end]

                                            if not valid_ts.empty:
                                                nearest_ts      = valid_ts.max()
                                                chart_end_index = int(ts_series[ts_series == nearest_ts].index[0])
                                            else:
                                                chart_end_index = len(ts_series) - 1

                                            # updating chart_position.
                                            nav = self.states.chart_navigation
                                            nav[crypto] = chart_end_index
                                            self.states.chart_navigation = nav
                                            st.rerun()

                                    with col_val:
                                        result = row["result"]
                                        change = row["change"]
                                        if result == "win":
                                            # Simple inline color cue
                                            st.markdown(f"{result}, +{change}")
                                        if result == "loss":
                                            st.markdown(f"{result}, {change}")

                                        

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
                        fig = self.create_chart(crypto, self.selected_indicators, highlighted_timestamps, current_position)


                        simulator = TradeSimulator(fig, crypto, self.df[crypto], self.states)

                        ### hovering- indicator data.
                        data_box = st.empty()               # the read-out panel
                        df_plot  = self.df[crypto]          # alias used by the callback

                        col1, col2 = st.columns([6, 1])

                        if fig:
                            with col1:
                                st.plotly_chart(
                                fig,
                                key=f"{crypto}_chart",                # must be unique per chart
                                use_container_width=True,
                                config=dict
                                    (
                                    displayModeBar=True,
                                    scrollZoom=True,
                                    dragmode="pan",
                                    modeBarButtonsToAdd=["drawline", "eraseshape"],
                                    modeBarButtonsToRemove=[
                                        'downloadPlot','toImage','lasso2d','select2d',
                                        'zoom2d','zoomIn2d','zoomOut2d','autoScale2d'
                                    ],
                                    ),
                                )
                                
                            with col2:
                                simulator.render()

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
                fig = self.create_chart(crypto, self.selected_indicators, highlighted_timestamps, current_position)
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

    def _update_price_yaxis_for_window(self, fig, df, start_time, end_time, *, pad=0.06):
        """Set y-axis range on the PRICE subplot (row=1,col=1) to the visible x-window."""
        if start_time is None or end_time is None or df is None or df.empty:
            return

        # slice the visible window (inclusive)
        mask = (df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)
        window = df.loc[mask]

        if window.empty:
            return

        y_min = float(window["low"].min())
        y_max = float(window["high"].max())

        # avoid zero-height ranges
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            return
        if y_max <= y_min:
            eps = max(1e-6, abs(y_max) * 1e-4)
            y_min -= eps
            y_max += eps

        # padding so markers / stars don't get cut off
        span = y_max - y_min
        y_min -= span * pad
        y_max += span * pad

        fig.update_yaxes(range=[y_min, y_max], row=1, col=1)

    def _add_auto_panning(self, fig, df, chart_position, highlighted_timestamps, timeframe, crypto, start_time=None):

        """Auto-pan priority: Navigation > chart_end (hit) > Highlights > None.
        NOTE: chart_end is NOT cleared; nav simply takes precedence when present.
        """

        # --- 1) Manual navigation (Back/Forward) takes priority ---
        if chart_position is not None and 0 <= chart_position < len(df):
            end_time = df.iloc[chart_position]['timestamp']
            self.states.chart_end = end_time
            start_time = self._calculate_window_start(end_time, timeframe, df)
            fig.update_xaxes(range=[start_time, end_time], type='date')
            self._update_price_yaxis_for_window(fig, df, start_time, end_time)
            # Do NOT clear chart_end; it remains stored but ignored while nav is active.
            return

        # if chart_positions not assigned or outside.
        if self.states.chart_end is not None:
            print("chart end exists!.")
            end_time = self.states.chart_end

            # Snap to an actual candle at or before end_time
            ts_series = df['timestamp']
            if end_time in set(ts_series):
                nav_idx = int(ts_series[ts_series == end_time].index[0])
            else:
                le = ts_series[ts_series <= end_time]
                nav_idx = int(le.index.max()) if not le.empty else len(df) - 1
                end_time = df.iloc[nav_idx]['timestamp']

            # Keep navigation pointer consistent so Forward/Back works from here
            nav_positions = self.states.chart_navigation
            nav_positions[crypto] = nav_idx
            self.states.chart_navigation = nav_positions

            start_time = self._calculate_window_start(end_time, timeframe, df)
            fig.update_xaxes(range=[start_time, end_time], type='date')
            self._update_price_yaxis_for_window(fig, df, start_time, end_time)
            return

        # if nav & chart_end inactive
        if highlighted_timestamps:
            print("highlighted_timestamps found!")
            prev_highlights_key = f"prev_highlighted_candles_{crypto}"
            prev_highlights = st.session_state.get(prev_highlights_key, [])

            latest_highlight = max(highlighted_timestamps)

            # Find matching (or closest earlier) candle
            matching_rows = df[df['timestamp'] == latest_highlight]
            if matching_rows.empty:
                earlier = df[df['timestamp'] <= latest_highlight]['timestamp']
                if not earlier.empty:   
                    closest = earlier.max()
                    matching_rows = df[df['timestamp'] == closest]

            if not matching_rows.empty:
                highlight_index = int(matching_rows.index[0])
                highlight_time = df.iloc[highlight_index]['timestamp']

                # Pan to the latest highlight if it's new; otherwise still pan (since nothing else is active)
                if len(highlighted_timestamps) > len(prev_highlights) or True:
                    start_time = self._calculate_window_start(highlight_time, timeframe, df)
                    fig.update_xaxes(range=[start_time, highlight_time], type='date')

                    # Sync nav pointer with highlight
                    nav_positions = self.states.chart_navigation
                    nav_positions[crypto] = highlight_index
                    self.states.chart_navigation = nav_positions

                    st.session_state[prev_highlights_key] = highlighted_timestamps.copy()
                    return
            else:
                # Clear invalid highlights and fall through
                highlights = self.states.highlighted_candles
                if crypto in highlights:
                    highlights[crypto] = []
                    self.states.highlighted_candles = highlights
                if prev_highlights_key in st.session_state:
                    del st.session_state[prev_highlights_key]

        # --- 4) Nothing to do ---
        return

    def _calculate_window_start(self, end_time, timeframe, df):
        """Calculate the start time for the chart window"""
        timeframe_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720, '1d': 1440
        }
        
        tf_minutes = timeframe_minutes.get(timeframe, 60)
        # Change how many candles you wanna see here.
        time_window_minutes = tf_minutes * 150  # Show ~60 candles
        start_time = end_time - pd.Timedelta(minutes=time_window_minutes)
        
        # Ensure start_time is not before data range
        data_start = df['timestamp'].min()
        if start_time < data_start:
            start_time = data_start
        
        return start_time
    
    def _add_unified_hover_overlay(self, fig, df, selected_inds, *, row=1, col=1, mute_candle=True):
        """
        Adds an invisible Scatter on the price subplot to provide a single rich hover box
        (works even if Candlestick lacks hovertemplate). Uses exact column names you provided.
        """

        # --- Build the list of columns to ship via customdata (order matters)
        custom_cols = []

        # Always include OHLC
        base_cols = ["open", "high", "low", "close", "chg", "chg_pct"]
        for c in base_cols:
            if c in df.columns:
                custom_cols.append(c)

        # RSI
        if "RSI" in selected_inds and "rsi" in df.columns:
            custom_cols.append("rsi")

        # Bollinger Band
        if "Bollinger Band" in selected_inds:
            for c in ["bb_middle", "bb_upper", "bb_lower"]:
                if c in df.columns:
                    custom_cols.append(c)

        # Stochastic sets
        if "Stochastic" in selected_inds:
            for c in ["stoch_k", "stoch_d"]:
                if c in df.columns: custom_cols.append(c)
        if "Stochastic2" in selected_inds:
            for c in ["stoch_k2", "stoch_d2"]:
                if c in df.columns: custom_cols.append(c)
        if "Stochastic3" in selected_inds:
            for c in ["stoch_k3", "stoch_d3"]:
                if c in df.columns: custom_cols.append(c)
        if "Stochastic4" in selected_inds:
            for c in ["stoch_k4", "stoch_d4"]:
                if c in df.columns: custom_cols.append(c)

        # KDJ (%K, %D, %J)
        if "KDJ" in selected_inds:
            for c in ["%K", "%D", "%J"]:
                if c in df.columns: custom_cols.append(c)

        # Williams % Range (WR)
        if "William % Range" in selected_inds and "WR" in df.columns:
            custom_cols.append("WR")

        # Build customdata (N x M)
        cols_data = []
        for c in custom_cols:
            # force numeric; strings or None -> NaN
            s = pd.to_numeric(df[c], errors="coerce")
            cols_data.append(s.to_numpy(dtype=float))
        customdata = np.column_stack(cols_data) if cols_data else None

        idx = {c: i for i, c in enumerate(custom_cols)}

        def val(col, fmt):
            # return a Plotly hovertemplate token like %{customdata[3]:.2f}
            if col in idx:
                return f"%{{customdata[{idx[col]}]:{fmt}}}"   # <-- note the ] before :
            return "n/a"
        # --- Build the hover box lines
        lines = [
            f"Open: {val('open', '.1f')}",
            f"High: {val('high', '.1f')}",
            f"Low: {val('low',  '.1f')}",
            f"Close: {val('close', '.1f')}"
            ]
        
        if "chg" in idx and "chg_pct" in idx:
            # sign on absolute, sign on percent; show % symbol
            lines.append(f"Change: {val('chg_pct', '+.2f')}%")
            
        # RSI
        if "RSI" in selected_inds and "rsi" in idx:
            lines.append(f"RSI: {val('rsi', '.2f')}")

        # Bollinger Band
        if "Bollinger Band" in selected_inds:
            parts = []
            if "bb_middle" in idx: parts.append(f"M:{val('bb_middle', '.4f')}")
            if "bb_upper"  in idx: parts.append(f"U:{val('bb_upper',  '.4f')}")
            if "bb_lower"  in idx: parts.append(f"L:{val('bb_lower',  '.4f')}")
            if parts: lines.append("BB: " + "  ".join(parts))

        # Stochastic
        if "Stochastic" in selected_inds:
            parts = []
            # if "stoch_k" in idx: parts.append(f"%K:{val('stoch_k', '.2f')}")
            if "stoch_d" in idx: parts.append(f"%D:{val('stoch_d', '.2f')}")
            if parts: lines.append("Stoch: " + "  ".join(parts))

        if "Stochastic2" in selected_inds:
            parts = []
            # if "stoch_k2" in idx: parts.append(f"%K:{val('stoch_k2', '.2f')}")
            if "stoch_d2" in idx: parts.append(f"%D:{val('stoch_d2', '.2f')}")
            if parts: lines.append("Stoch2: " + "  ".join(parts))

        if "Stochastic3" in selected_inds:
            parts = []
            # if "stoch_k3" in idx: parts.append(f"%K:{val('stoch_k3', '.2f')}")
            if "stoch_d3" in idx: parts.append(f"%D:{val('stoch_d3', '.2f')}")
            if parts: lines.append("Stoch3: " + "  ".join(parts))

        if "Stochastic4" in selected_inds:
            parts = []
            # if "stoch_k4" in idx: parts.append(f"%K:{val('stoch_k4', '.2f')}")
            if "stoch_d4" in idx: parts.append(f"%D:{val('stoch_d4', '.2f')}")
            if parts: lines.append("Stoch4: " + "  ".join(parts))

        # KDJ
        if "KDJ" in selected_inds:
            parts = []
            if "%K" in idx: parts.append(f"%K:{val('%K', '.2f')}")
            if "%D" in idx: parts.append(f"%D:{val('%D', '.2f')}")
            if "%J" in idx: parts.append(f"%J:{val('%J', '.2f')}")
            if parts: lines.append("KDJ: " + "  ".join(parts))

        # Williams % Range
        if "William % Range" in selected_inds and "WR" in idx:
            lines.append(f"W%R: {val('WR', '.2f')}")

        hovertemplate = "<br>".join(lines) + "<extra></extra>"

        # Add the invisible overlay on the PRICE subplot
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["close"],                # aligns to candles for hover
                customdata=customdata,
                mode="markers",
                marker=dict(size=8, opacity=0),  # invisible but hoverable
                hovertemplate=hovertemplate,
                name="_hover_",
                showlegend=False,
                hoverlabel=dict(align="left"),
            ),
            row=row, col=col
        )

        # Optional: silence the candlestick’s own hover to avoid duplicates
        if mute_candle:
            for tr in fig.data:
                if isinstance(tr, go.Candlestick):
                    tr.update(hoverinfo="skip")
        
    def _yref_for_row(self, row_idx: int) -> str:
        """
        Plotly yref for row index (1-based): row 1 -> 'y', row 2 -> 'y2', etc.
        """
        return 'y' if row_idx == 1 else f'y{row_idx}'

    def _yaxis_name(self, row_idx: int) -> str:
    # row 1 -> 'yaxis', row 2 -> 'yaxis2', ...
        return 'yaxis' if row_idx == 1 else f'yaxis{row_idx}'

    def _row_center_in_paper(self, fig, row_idx: int) -> float:
        # Use the subplot's actual domain from plotly
        ya = self._yaxis_name(row_idx)
        dom = getattr(fig.layout, ya).domain  # [y0, y1] in paper coords
        return (dom[0] + dom[1]) / 2.0


    def _add_right_edge_labels(self, fig, crypto, sep_inds):
        """
        Put compact value badges OUTSIDE the plot area, vertically centered per subplot.
        Compatible with custom row_heights.
        """
        df = self.df[crypto]
        if df is None or df.empty:
            return

        # Space for labels in the margin
        fig.update_layout(margin=dict(r=100))

        # Helper to get latest non-NaN
        def latest(col):
            if col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce")
            idx = s.last_valid_index()
            return None if idx is None else float(s.loc[idx])

        # Which row an indicator is on
        def row_for_indicator(ind_name: str) -> int:
            return 1 if ind_name not in sep_inds else sep_inds.index(ind_name) + 2

        # Common kwargs: outside the plot, anchored to right edge, shifted into margin
        def add_box(row_idx: int, text: str):
            fig.add_annotation(
                xref="paper", yref="paper",
                x=1.0, y=self._row_center_in_paper(fig, row_idx),
                xanchor="left", yanchor="middle",
                xshift=12,  # push into the right margin
                text=text,
                showarrow=False,
                bgcolor="#1e222a",
                bordercolor="#888", borderwidth=1, borderpad=4,
                font=dict(color="white", size=12),
                align="left",
            )

        # ----- Price (row 1)
        close_val = latest("close")
        if close_val is not None:
            add_box(1, f"${close_val:.1f}")

        # ----- Indicators
        for ind in self.selected_indicators:
            row = row_for_indicator(ind)

            if ind == "RSI":
                v = latest("rsi")
                if v is not None: add_box(row, f"RSI {v:.2f}")

            elif ind == "Bollinger Band":
                parts = []
                for label, col in (("U", "bb_upper"), ("M", "bb_middle"), ("L", "bb_lower")):
                    v = latest(col)
                    if v is not None: parts.append(f"{label}:{v:.4f}")
                if parts: add_box(row, "BB " + " ".join(parts))

            elif ind.startswith("Stochastic"):
                suffix = "" if ind == "Stochastic" else ind.replace("Stochastic", "")
                v = latest("stoch_d" + suffix)
                if v is not None: add_box(row, f"%D {v:.2f}")

            elif ind == "KDJ":
                parts = []
                for label, col in (("%K", "%K"), ("%D", "%D"), ("%J", "%J")):
                    v = latest(col)
                    if v is not None: parts.append(f"{label}:{v:.2f}")
                if parts: add_box(row, " ".join(parts))

            elif ind == "William % Range":
                v = latest("WR")
                if v is not None: add_box(row, f"W%R {v:.2f}")

    def create_chart(self, crypto, indicators, highlighted_timestamps=None, chart_position=None) -> go.Figure:
        """Create OHLC candlestick chart with optional highlighting and navigation"""
        if self.df[crypto] is None or self.df[crypto].empty:
            return None
                
        sep_inds = []
        for ind in self.selected_indicators:
            if ind in SEPARATE_AX_INDICATORS:
                sep_inds.append(ind)
        
        # Changes height of chart as more indicators are added.
        n_rows = 1 + len(sep_inds)
        weights = [5] + [1] * len(sep_inds)
        total_weight = sum(weights)
        row_heights = [w/total_weight for w in weights]

        df_local = self.df[crypto]
        if "chg" not in df_local.columns or "chg_pct" not in df_local.columns:
            prev_close = pd.to_numeric(df_local["close"], errors="coerce").shift(1)
            close_now = pd.to_numeric(df_local["close"], errors="coerce")
            chg = close_now - prev_close
            # avoid divide-by-zero
            chg_pct = np.where(prev_close != 0, round((chg / prev_close) * 100.0, 2), np.nan)
            self.df[crypto]["chg"] = chg
            self.df[crypto]["chg_pct"] = chg_pct

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
        
        # Map indicator names -> (constructor, params_key)
        name_map = {
            "RSI": (RSIIndicator, "RSI"),
            "Bollinger Band": (BollingerBandsIndicator, "Bollinger Band"),
            "KDJ": (KDJIndicator, "KDJ"),
            "Half Trend": (HalfTrendIndicator, "Half Trend"),
            "William % Range": (WilliamsRIndicator, "William % Range"),

            # Stochastic variants (all use StochasticIndicator, keyed separately)
            "Stochastic":  (StochasticIndicator, "Stochastic"),
            "Stochastic2": (StochasticIndicator, "Stochastic2"),
            "Stochastic3": (StochasticIndicator, "Stochastic3"),
            "Stochastic4": (StochasticIndicator, "Stochastic4"),
        }

        for ind in self.selected_indicators:
            entry = name_map.get(ind)
            if not entry:
                continue

            ctor, params_key = entry
            params = dict(self.indicator_params.get(params_key, {}))  # copy so we can modify

            # Inject suffix / plotting defaults for Stochastic variants
            if ctor is StochasticIndicator:
                # "", "2", "3", "4"
                suffix = "" if ind == "Stochastic" else ind.replace("Stochastic", "")
                params.setdefault("suffix", suffix)

            indicator = ctor(**params)

            # Optional: give each instance a distinct legend/display name
            try:
                indicator.name = ind
            except Exception:
                pass

            # Calculate & plot
            self.df[crypto] = indicator.calculate(self.df[crypto])
            indicator.add_traces(
                fig,
                self.df[crypto],
                1 if ind not in sep_inds else sep_inds.index(ind) + 2
            )
        
        # this adds indicator values inside the black box when you hover over candlesticks
        self._add_unified_hover_overlay(fig, self.df[crypto], self.selected_indicators, row=1, col=1)
        self._add_right_edge_labels(fig, crypto, sep_inds)

        fig.update_layout(hovermode="x unified")


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

        """ Code for shared hover feature(NOT WORKING YET)"""
        # keep ranges synced across rows
        # unify hover and crosshair behavior
        fig.update_xaxes(matches="x")
        fig.update_layout(
            hoversubplots="axis",          # include stacked subplots sharing the x-axis
            hovermode="x unified",
        )

        fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor")

        if chart_position is not None or highlighted_timestamps or self.states.chart_end is not None:
            print(f"chart_position is {chart_position}, highlighted_timestamps is {highlighted_timestamps}, backtest is {self.states.bt_mode} and chart_end is {self.states.chart_end}")
            self._add_auto_panning(fig, self.df[crypto], chart_position, highlighted_timestamps, self.timeframe, crypto)
            

        return fig