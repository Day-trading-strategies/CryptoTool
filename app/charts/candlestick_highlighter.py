# app/charts/candlestick_highlighter.py
import streamlit as st
import pandas as pd

class CandlestickHighlighter:
    """Handles candlestick highlighting functionality"""
    
    def __init__(self, crypto, df, states):
        self.crypto = crypto
        self.df = df
        self.states = states
        self.highlight_key = f"highlighted_candles_{crypto}"
        self.tab_idx = None  # Will be set if used in tabs
    
    def render(self):
        """Render highlighting controls"""
        st.markdown("**🎯 Highlight Candlesticks:**")
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        # Create unique keys for controls
        key_suffix = f"_{self.crypto}"
        if self.tab_idx is not None:
            key_suffix += f"_{self.tab_idx}"
        
        with col1:
            selected_timestamp = self._render_datetime_picker(key_suffix)
        
        with col2:
            if st.button("⭐ Highlight", key=f"highlight{key_suffix}"):
                self._add_highlight(selected_timestamp)
        
        with col3:
            if st.button("❌ Remove", key=f"remove{key_suffix}"):
                self._remove_highlight(selected_timestamp)
        
        with col4:
            if st.button("🗑️ Clear All", key=f"clear_all{key_suffix}"):
                self._clear_all_highlights()
        
        self._show_current_highlights()
        
        # Handle auto-pan to new highlights
        self._handle_auto_pan_to_highlights()
        
        return self._get_highlighted_timestamps()
    
    def _render_datetime_picker(self, key_suffix):
        """Render date and time picker"""
        min_date = self.df['timestamp'].min().date()
        max_date = self.df['timestamp'].max().date()
        
        selected_date = st.date_input(
            "Select Date:",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key=f"date{key_suffix}"
        )
        
        available_times = self.df[self.df['timestamp'].dt.date == selected_date]['timestamp'].dt.time.tolist()
        
        if available_times:
            selected_time = st.selectbox(
                "Select Time:",
                options=available_times,
                index=len(available_times)-1,
                format_func=lambda x: x.strftime('%H:%M:%S'),
                key=f"time{key_suffix}"
            )
            
            selected_datetime = pd.Timestamp.combine(selected_date, selected_time)
            
            if self.df[self.df['timestamp'] == selected_datetime].empty:
                st.warning("No candlestick found for selected date/time")
                return None
            else:
                st.info(f"Selected: {selected_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                return selected_datetime
        else:
            st.warning("No data available for selected date")
            return None
    
    def _get_highlighted_timestamps(self):
        """Get list of highlighted timestamps for this crypto"""
        highlights = self.states.highlighted_candles
        return highlights.get(self.crypto, [])
    
    def _add_highlight(self, timestamp):
        """Add a timestamp to highlights"""
        if timestamp is not None:
            highlights = self.states.highlighted_candles
            if self.crypto not in highlights:
                highlights[self.crypto] = []
            if timestamp not in highlights[self.crypto]:
                highlights[self.crypto].append(timestamp)
                self.states.highlighted_candles = highlights
                st.rerun()
    
    def _remove_highlight(self, timestamp):
        """Remove a timestamp from highlights"""
        if timestamp is not None:
            highlights = self.states.highlighted_candles
            if self.crypto in highlights and timestamp in highlights[self.crypto]:
                highlights[self.crypto].remove(timestamp)
                self.states.highlighted_candles = highlights
                self._update_previous_highlights()
                st.rerun()
    
    def _clear_all_highlights(self):
        """Clear all highlights for this crypto"""
        highlights = self.states.highlighted_candles
        if self.crypto in highlights and highlights[self.crypto]:
            highlights[self.crypto] = []
            self.states.highlighted_candles = highlights
            self._update_previous_highlights()
            st.rerun()
    
    def _update_previous_highlights(self):
        """Update the previous highlights state"""
        prev_highlights = self.states.previous_highlights
        current_highlights = self.states.highlighted_candles
        prev_highlights[self.crypto] = current_highlights.get(self.crypto, []).copy()
        self.states.previous_highlights = prev_highlights
    
    def _handle_auto_pan_to_highlights(self):
        """Handle auto-pan logic when new highlights are added"""
        highlighted_timestamps = self._get_highlighted_timestamps()
        
        if highlighted_timestamps:
            prev_highlights_key = f"prev_highlighted_candles_{self.crypto}"
            prev_highlights = st.session_state.get(prev_highlights_key, [])
            
            # If we have new highlights, auto-pan to the latest one
            if len(highlighted_timestamps) > len(prev_highlights):
                latest_highlight = max(highlighted_timestamps)
                
                # Find the closest matching row in the dataframe
                matching_rows = self.df[self.df['timestamp'] == latest_highlight]
                if matching_rows.empty:
                    # Find closest earlier timestamp
                    earlier_timestamps = self.df[self.df['timestamp'] <= latest_highlight]['timestamp']
                    if not earlier_timestamps.empty:
                        closest_timestamp = earlier_timestamps.max()
                        matching_rows = self.df[self.df['timestamp'] == closest_timestamp]
                
                if not matching_rows.empty:
                    # Update navigation position to the highlighted candlestick
                    actual_timestamp = matching_rows.iloc[0]['timestamp']
                    position_rows = self.df[self.df['timestamp'] == actual_timestamp]
                    if not position_rows.empty:
                        highlight_position = int(position_rows.index[0])
                        nav_positions = self.states.chart_navigation
                        nav_positions[self.crypto] = highlight_position
                        self.states.chart_navigation = nav_positions
                
                # Update the previous highlights state
                st.session_state[prev_highlights_key] = highlighted_timestamps.copy()
    
    def _show_current_highlights(self):
        """Display current highlights"""
        highlighted_timestamps = self._get_highlighted_timestamps()
        
        if highlighted_timestamps:
            highlight_info = []
            for ts in highlighted_timestamps:
                exact_match = self.df[self.df['timestamp'] == ts]
                if not exact_match.empty:
                    highlight_info.append(f"{ts.strftime('%H:%M:%S')}")
                else:
                    earlier_timestamps = self.df[self.df['timestamp'] <= ts]['timestamp']
                    if not earlier_timestamps.empty:
                        closest_timestamp = earlier_timestamps.max()
                        highlight_info.append(f"{ts.strftime('%H:%M:%S')}→{closest_timestamp.strftime('%H:%M:%S')}")
            
            st.info(f"💫 Currently highlighted: {sorted(highlight_info)}")
            if any('→' in info for info in highlight_info):
                st.caption("🔶 Orange stars show approximate matches (closest earlier time)")