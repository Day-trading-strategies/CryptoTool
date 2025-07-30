import streamlit as st
from datetime import datetime, timedelta

from app.back_tester import CryptoBacktester
from app.config import *

class BacktestSettings:
    """Handles Backtesting conditions expander"""
    def __init__(self, selected_indicators, states):
        self.selected_crypto = None
        self.start_date = None
        self.end_date = None
        self.selected_indicators = selected_indicators
        self.states = states
        self.render()

    def render(self):
        """Render the sidebar components"""
        with st.expander("Backtesting Settings"):
            self.render_dates()
            self.render_crypto()
            self.render_conditions()

    def render_dates(self):
        self.start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=30),
            key="bt_start"
        )
        self.end_date = st.date_input(
            "End Date",
            value=datetime.now().date(),
            key="bt_end"
        )

    def render_crypto(self):
        crypto = st.selectbox(
            "Select Crypto:",
            options=list(AVAILABLE_CRYPTOS.keys()),
            index=0  # Default to BTC
            )
        self.states.crypto = crypto
        
    def render_conditions(self):
        st.markdown("**Indicator Conditions(select indicator in sidebar to show.)**")
        
        indicator_conditions = {}
        if "RSI" in self.selected_indicators:
            indicator_conditions["RSI_U"] = st.number_input("RSI Upper Bound", 0.0, 100.0, None, step=0.1, key="bt_rsi_u")
            indicator_conditions["RSI_L"] = st.number_input("RSI Lower Bound", 0.0, 100.0, None, step=0.1, key="bt_rsi_l")

        # William %R conditions
        if "William % Range" in self.selected_indicators:
            indicator_conditions["WR_U"] = st.number_input(
                    "William %R Upper Bound(e.g. -20", -100.0, 0.0, None, step=0.1, key="bt_wr_u"
                )
            indicator_conditions["WR_L"] = st.number_input(
                    "William %R Lower Bound(e.g. -80)", -100.0, 0.0, None, step=0.1, key="bt_wr_l"
                )
        # Half Trend
        if "Half Trend" in self.selected_indicators:
            ht_cond = st.selectbox(
                "Half Trend Conditions",
                options=[
                    "Buy Signal",
                    "Sell Signal"],
                index=0
                )
            if ht_cond == "Buy Signal":
                indicator_conditions["ht_buy"] = True
            else:
                indicator_conditions["ht_sell"] = True

        if "Bollinger Band" in self.selected_indicators:
            # Bollinger Band
            if st.checkbox('Price "High" Touches Top BB'):
                indicator_conditions["Bollinger Top"] = True
            if st.checkbox('Price "Low" Touches Bottom BB'):
                indicator_conditions["Bollinger Bottom"] = True
            if st.checkbox("Price Touches Either"):
                indicator_conditions["Bollinger Either"] = True

        
        # KDJ conditions
        if "KDJ" in self.selected_indicators:
            if st.checkbox("KDJ Intersection", key="bt_chk_kdj"):
                indicator_conditions["KDJ"] = True
        

        run_bt = st.button("▶ Run Backtest", key="bt_run")

        if run_bt:
            self.states.bt_mode = True
            self.states.ob = CryptoBacktester(
                            self.start_date,
                            self.end_date,
                            indicator_conditions,
                            self.states)
            # st.rerun()

