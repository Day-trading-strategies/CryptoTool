# app/charts/chart_navigation.py
import streamlit as st
import pandas as pd

class ChartNavigation:
    """Handles chart navigation controls"""
    
    def __init__(self, crypto, df, states):
        self.crypto = crypto
        self.df = df
        self.states = states
        self.nav_key = f"chart_nav_{crypto}"
        self.timeframe_key = f"timeframe_{crypto}"
        self.tab_idx = None  # Will be set if used in tabs
        
    def render(self):
        """Render navigation controls"""
        # Handle timeframe changes first
        self._handle_timeframe_changes()
        current_position = self._get_current_position()
        st.markdown("**📊 Chart Navigation:**")
        
        col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
        
        # Create unique button keys
        key_suffix = f"_{self.crypto}"
        if self.tab_idx is not None:
            key_suffix += f"_{self.tab_idx}"
        
        with col1:
            if st.button("⬅️ Back", 
                        key=f"nav_back{key_suffix}", 
                        disabled=current_position <= 0,
                        use_container_width=True):
                self._navigate_back()
        
        with col2:
            if st.button("➡️ Forward", 
                        key=f"nav_forward{key_suffix}", 
                        disabled=current_position >= len(self.df) - 1,
                        use_container_width=True):
                self._navigate_forward()
        
        with col3:
            current_candle = self.df.iloc[current_position]
            st.info(f"📍 Position: {current_position + 1}/{len(self.df)} | Time: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col4:
            if st.button("🏠 Latest", 
                        key=f"nav_latest{key_suffix}", 
                        disabled=current_position >= len(self.df) - 1,
                        use_container_width=True):
                self.navigate_to_latest()
                st.rerun()
        
        return current_position
    
    def _handle_timeframe_changes(self):
        """Handle timeframe changes by resetting navigation"""
        current_timeframe = st.session_state.get('selected_timeframe', '15m')
        previous_timeframe = st.session_state.get(self.timeframe_key, None)
        print("entered _handle_timeframe_changes")

        if previous_timeframe != current_timeframe:
            nav_positions = self.states.chart_navigation
            print("prev timeframe != current timeframe")

            # If a backtest highlight exists, jump there; otherwise go to latest candle
            highlight_ts = getattr(self.states, 'chart_end', None)
            if highlight_ts is not None:
                print("backtest highlight found")
                # find the closest timestamp ≤ highlight_ts in the new interval DF
                ts_series = self.df['timestamp']
                valid_ts  = ts_series[ts_series <= highlight_ts]
                if not valid_ts.empty:
                    nearest_ts = valid_ts.max()
                    idx        = int(ts_series[ts_series == nearest_ts].index[0])
                else:
                    idx        = len(self.df) - 1
                nav_positions[self.crypto] = idx
                print("backtest highlight not found")
            else:
                nav_positions[self.crypto] = len(self.df) - 1

            self.states.chart_navigation = nav_positions

            # Clear old auto-pan state so highlight logic re-triggers
            prev_highlights_key = f"prev_highlighted_candles_{self.crypto}"
            if prev_highlights_key in st.session_state:
                del st.session_state[prev_highlights_key]

            # Record this new timeframe
            st.session_state[self.timeframe_key] = current_timeframe
    
    def _get_current_position(self):
        """Get current navigation position"""
        nav_positions = self.states.chart_navigation
        if self.crypto not in nav_positions or (nav_positions[self.crypto] < 0 or nav_positions[self.crypto] >= len(self.df)):
            nav_positions[self.crypto] = len(self.df) - 1
            self.states.chart_navigation = nav_positions
        return int(nav_positions[self.crypto])  # Ensure it's a Python int
    
    def _navigate_back(self):
        """Navigate to previous candlestick"""
        nav_positions = self.states.chart_navigation
        if nav_positions[self.crypto] > 0:
            nav_positions[self.crypto] = int(nav_positions[self.crypto] - 1)
            self.states.chart_navigation = nav_positions
            st.rerun()
    
    def _navigate_forward(self):
        """Navigate to next candlestick"""
        nav_positions = self.states.chart_navigation
        if nav_positions[self.crypto] < len(self.df) - 1:
            nav_positions[self.crypto] = int(nav_positions[self.crypto] + 1)
            self.states.chart_navigation = nav_positions
            st.rerun()
    
    def navigate_to_latest(self):
        """Navigate to latest candlestick"""
        nav_positions = self.states.chart_navigation
        nav_positions[self.crypto] = int(len(self.df) - 1)
        self.states.chart_navigation = nav_positions
        