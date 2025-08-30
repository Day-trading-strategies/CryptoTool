import streamlit as st

from app.config import *

class Sidebar:
    """Handles sidebar controls and user input interaction"""

    def __init__(self):
        self.selected_cryptos = []
        self.selected_indicator = []
        self.selected_timeframe = '1h'
        self.indicator_params = {}
        self.render()
    
    def render(self):
        """Render the sidebar components"""
        self.render_headers()
        self.render_crypto()
        self.render_indicators()
        self.render_timeframe()
        self.render_refresh()
        self.render_indicator_params()

    def render_headers(self):
        # Sidebar controls
        st.sidebar.header("Settings")
        # Cryptocurrency selection
        st.sidebar.subheader("Select Cryptocurrencies")

    def render_crypto(self):
        self.selected_cryptos = st.sidebar.multiselect(
            "Choose cryptocurrencies to monitor:",
            options=list(AVAILABLE_CRYPTOS.keys()),
            default=DEFAULT_CRYPTOS
        )

    def render_indicators(self):
        #indicator selection
        st.sidebar.subheader("Indicator")
        self.selected_indicator = sorted(st.sidebar.multiselect(
            "Select Indicator:",
            options=AVAILABLE_INDICATORS,
            default=DEFAULT_INDICATORS
        ))
        

    def render_timeframe(self):
        st.sidebar.subheader("Timeframe")
        self.selected_timeframe = st.sidebar.selectbox(
        "Select timeframe:",
        options=list(TIMEFRAMES.keys()),
        index=3  # Default to 1h
    )
        
    def render_refresh(self):
        # Refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()
        st.sidebar.markdown("---")

    def render_indicator_params(self):
        if "RSI" in self.selected_indicator:
            self.indicator_params["RSI"] = {
                "window": st.sidebar.number_input("RSI window", 2, 100, 14, key="RSI_w")
            }

        if "Bollinger Band" in self.selected_indicator:
            self.indicator_params["Bollinger Band"] = {
                "window":     st.sidebar.number_input("BB window", 2, 100, 20, key="BB_w"),
                "window_dev": st.sidebar.slider("BB σ-dev", 1.0, 4.0, 2.0, 0.1, key="BB_d")
            }

        if "KDJ" in self.selected_indicator:
            self.indicator_params["KDJ"] = {
                "window": st.sidebar.number_input("KDJ window", 2, 100, 9, key="KDJ_p"),
                "smoothing": st.sidebar.number_input("KDJ smoothing period", 1, 10, 3, key="KDJ_s"),
            }

        if "William % Range" in self.selected_indicator:
            self.indicator_params["William % Range"] = {
                "lbp": st.sidebar.number_input("William %R window", 2, 100, 14, key="WR_w")
            }

        if "Half Trend" in self.selected_indicator:
            self.indicator_params["Half Trend"] = {
                "amplitude":     st.sidebar.number_input("HT Amplitude", 2, 50, 5, key="HT_p"),
            }

        if "Stochastic" in self.selected_indicator:
                with st.sidebar.expander(f"Stochastic parameters", expanded=False):

                    self.indicator_params["Stochastic"] = {
                        "k_window" : st.number_input(
                            "%K Window", 2, 100, 9, key=f"ST_w"
                        ),
                        "k_smoothing" : st.number_input(
                            "%K Smoothing", 1, 50, 1,  key=f"ST_ks"
                        ),
                        "d_smoothing" : st.number_input(
                            "%D Smoothing", 1, 50, 3,  key=f"ST_ds"
                        )
                    }
        if "Stochastic2" in self.selected_indicator:
                with st.sidebar.expander(f"Stochastic2 parameters", expanded=False):

                    self.indicator_params["Stochastic2"] = {
                        "k_window" : st.number_input(
                            "%K Window", 2, 100, 14, key=f"ST_w2"
                        ),
                        "k_smoothing" : st.number_input(
                            "%K Smoothing", 1, 50, 1,  key=f"ST_ks2"
                        ),
                        "d_smoothing" : st.number_input(
                            "%D Smoothing", 1, 50, 3,  key=f"ST_ds2"
                        )
                    }
        if "Stochastic3" in self.selected_indicator:
                with st.sidebar.expander(f"Stochastic3 parameters", expanded=False):

                    self.indicator_params["Stochastic3"] = {
                        "k_window" : st.number_input(
                            "%K Window", 2, 100, 40, key=f"ST_w3"
                        ),
                        "k_smoothing" : st.number_input(
                            "%K Smoothing", 1, 50, 3,  key=f"ST_ks3"
                        ),
                        "d_smoothing" : st.number_input(
                            "%D Smoothing", 1, 50, 4,  key=f"ST_ds3"
                        )
                    }
        if "Stochastic4" in self.selected_indicator:
                with st.sidebar.expander(f"Stochastic4 parameters", expanded=False):

                    self.indicator_params["Stochastic4"] = {
                        "k_window" : st.number_input(
                            "%K Window", 2, 100, 60, key=f"ST_w4"
                        ),
                        "k_smoothing" : st.number_input(
                            "%K Smoothing", 1, 50, 1,  key=f"ST_ks4"
                        ),
                        "d_smoothing" : st.number_input(
                            "%D Smoothing", 1, 50, 10,  key=f"ST_ds4"
                        )
                    }

        

                

                