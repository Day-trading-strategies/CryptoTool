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
        # self._handle_timeframe_changes()
        current_position = self._get_current_position()
        st.markdown("<div style='padding-top:0px'></div>", unsafe_allow_html=True)
        st.markdown("**📊 Chart Navigation:**")
        
        col1, col2, col3, col4 = st.columns([1,1,1,1])        
        # Create unique button keys
        key_suffix = f"_{self.crypto}"
        if self.tab_idx is not None:
            key_suffix += f"_{self.tab_idx}"
        
        with col4:
            if st.button("⬅️ 15", 
                        key=f"nav_back15{key_suffix}", 
                        disabled=current_position <= 0,
                        use_container_width=True):
                for _ in range(15):
                    self._navigate_back()
                st.rerun()
        with col3:
            if st.button("⬅️ 10 ", 
                        key=f"nav_back10{key_suffix}", 
                        disabled=current_position <= 0,
                        use_container_width=True):
                for _ in range(10):
                    self._navigate_back()
                st.rerun()
        
        with col2:
            if st.button("⬅️ 5", 
                        key=f"nav_back5{key_suffix}", 
                        disabled=current_position <= 0,
                        use_container_width=True):
                for _ in range(5):
                    self._navigate_back()
                st.rerun()
        
        with col1:
            if st.button("⬅️ 1", 
                        key=f"nav_back{key_suffix}", 
                        disabled=current_position <= 0,
                        use_container_width=True):
                self._navigate_back()
                st.rerun()
        col1, col2, col3, col4 = st.columns([1,1,1,1])

        with col1:
            if st.button("➡️ 1", 
                        key=f"nav_forward{key_suffix}", 
                        disabled=current_position >= len(self.df) - 1,
                        use_container_width=True):
                self._navigate_forward()
                st.rerun()
        
        with col2:
            if st.button("➡️ 5", 
                        key=f"nav_forward5{key_suffix}", 
                        disabled=current_position >= len(self.df) - 1,
                        use_container_width=True):
                for _ in range(5):
                    self._navigate_forward()
                st.rerun()
        
        with col3:
            if st.button("➡️ 10", 
                        key=f"nav_forward10{key_suffix}", 
                        disabled=current_position >= len(self.df) - 1,
                        use_container_width=True):
                for _ in range(10):
                    self._navigate_forward()
                st.rerun()
        
        with col4:
            if st.button("➡️ 15", 
                        key=f"nav_forward15{key_suffix}", 
                        disabled=current_position >= len(self.df) - 1,
                        use_container_width=True):
                for _ in range(15):
                    self._navigate_forward()
                st.rerun()
        
        col9, col10 = st.columns([2, 1])

        with col9:
            current_candle = self.df.iloc[current_position]
            st.info(f"📍 Position: {current_position + 1}/{len(self.df)} | Time: {current_candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col10:
            if st.button("🏠 Latest", 
                        key=f"nav_latest{key_suffix}", 
                        disabled=current_position >= len(self.df) - 1,
                        use_container_width=True):
                self.navigate_to_latest()
                st.rerun()
        
        return current_position
    
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
    
    def _navigate_forward(self):
        # print("forward navigation clicked")
        """Navigate to next candlestick"""
        nav_positions = self.states.chart_navigation
        if nav_positions[self.crypto] < len(self.df) - 1:
            nav_positions[self.crypto] = int(nav_positions[self.crypto] + 1)
            self.states.chart_navigation = nav_positions

    def navigate_to_latest(self):
        """Navigate to latest candlestick"""
        nav_positions = self.states.chart_navigation
        nav_positions[self.crypto] = int(len(self.df) - 1)
        self.states.chart_navigation = nav_positions
        