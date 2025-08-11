import streamlit as st
from datetime import datetime, timedelta

from app.backtest.backtester import CryptoBacktester
from app.indicators.half_trend import HalfTrendIndicator
from app.indicators.bollinger_bands import BollingerBandsIndicator
from app.indicators.rsi import RSIIndicator
from app.indicators.williams_r import WilliamsRIndicator
from app.indicators.kdj import KDJIndicator
from app.indicators.stochastic import StochasticIndicator
from app.config import *

class BacktestSettings:
    """Handles Backtesting conditions expander"""
    def __init__(self, selected_indicators, timeframe, states, data_fetcher, params):
        self.selected_crypto = None
        self.timeframe = timeframe
        self.start_date = None
        self.end_date = None
        self.selected_indicators = selected_indicators
        self.states = states
        self.data_fetcher = data_fetcher
        self.indicator_params = params
        self.render()

    def render(self):
        """Render the sidebar components"""
        with st.expander("Backtesting Settings"):
            self.render_dates()
            self.render_crypto()
            self.render_timeframe()
            self.render_conditions()

    def render_dates(self):
        self.start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=5),
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

    def render_timeframe(self):
        timeframe = st.selectbox(
            "Select Timeframe:",
            options=list(TIMEFRAMES.keys()),
            index=3 # Default to 1hr
        )
        self.states.timeframe = timeframe
        
    def render_conditions(self):
        st.markdown("**Indicator Conditions(select indicator in sidebar to show.)**")
        
        indicator_conditions = {}
        if "RSI" in self.selected_indicators:
            indicator_conditions["RSI_U"] = st.number_input("RSI Upper Bound", 0.0, 100.0, None, step=0.1, key="bt_rsi_u")
            indicator_conditions["RSI_L"] = st.number_input("RSI Lower Bound", 0.0, 100.0, None, step=0.1, key="bt_rsi_l")

        if "Stochastic" in self.selected_indicators:
            indicator_conditions["ST_U"] = st.number_input("Stoch Upper Bound", 0.0, 100.0, 80.0, step=0.1, key="bt_st_u")
            indicator_conditions["ST_L"] = st.number_input("Stoch Lower Bound", 0.0, 100.0, 20.0, step=0.1, key="bt_st_l")
        if "Stochastic2" in self.selected_indicators:
            indicator_conditions["ST_U2"] = st.number_input("Stoch2 Upper Bound", 0.0, 100.0, 80.0, step=0.1, key="bt_st_u2")
            indicator_conditions["ST_L2"] = st.number_input("Stoch2 Lower Bound", 0.0, 100.0, 20.0, step=0.1, key="bt_st_l2")
        if "Stochastic3" in self.selected_indicators:
            indicator_conditions["ST_U3"] = st.number_input("Stoch3 Upper Bound", 0.0, 100.0, 80.0, step=0.1, key="bt_st_u3")
            indicator_conditions["ST_L3"] = st.number_input("Stoch3 Lower Bound", 0.0, 100.0, 20.0, step=0.1, key="bt_st_l3")
        if "Stochastic4" in self.selected_indicators:
            indicator_conditions["ST_U4"] = st.number_input("Stoch4 Upper Bound", 0.0, 100.0, 80.0, step=0.1, key="bt_st_u4")
            indicator_conditions["ST_L4"] = st.number_input("Stoch4 Lower Bound", 0.0, 100.0, 20.0, step=0.1, key="bt_st_l4")

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
            print(f"self.selected_indicators is {self.selected_indicators}")

            self.states.bt_mode = True
            self.states.ob = CryptoBacktester(
                            self.start_date,
                            self.end_date,
                            indicator_conditions,
                            self.states)

            progress_text = "Loading Data... Please Wait"
            my_bar = st.progress(0, text=progress_text)
            # fetches data to do backtest on.
            df = self.data_fetcher.fetch_ohlc_data_range(
                AVAILABLE_CRYPTOS[self.states.crypto], 
                self.states.timeframe, self.states.ob.start_date, self.states.ob.end_date
                )
            # Fetches data to store for faster transition between timeframes
            my_bar.progress(30)
            self.data_fetcher.fetch_ohlc_data_range(
                AVAILABLE_CRYPTOS[self.states.crypto], 
                '3m', self.states.ob.start_date, self.states.ob.end_date
                ).to_csv("data/3m_df.csv", index=False)
            my_bar.progress(45)
            self.data_fetcher.fetch_ohlc_data_range(
                AVAILABLE_CRYPTOS[self.states.crypto], 
                '5m', self.states.ob.start_date, self.states.ob.end_date
                ).to_csv("data/5m_df.csv", index=False)
            my_bar.progress(55)
            self.data_fetcher.fetch_ohlc_data_range(
                AVAILABLE_CRYPTOS[self.states.crypto], 
                '15m', self.states.ob.start_date, self.states.ob.end_date
                ).to_csv("data/15m_df.csv", index=False)
            my_bar.progress(60)
            self.data_fetcher.fetch_ohlc_data_range(
                AVAILABLE_CRYPTOS[self.states.crypto], 
                '1h', self.states.ob.start_date, self.states.ob.end_date
                ).to_csv("data/1h_df.csv", index=False)
            self.data_fetcher.fetch_ohlc_data_range(
                AVAILABLE_CRYPTOS[self.states.crypto], 
                '4h', self.states.ob.start_date, self.states.ob.end_date
                ).to_csv("data/4h_df.csv", index=False)
            self.data_fetcher.fetch_ohlc_data_range(
                AVAILABLE_CRYPTOS[self.states.crypto], 
                '1d', self.states.ob.start_date, self.states.ob.end_date
                ).to_csv("data/1d_df.csv", index=False)
            # self.data_fetcher.fetch_ohlc_data_range(
            #     AVAILABLE_CRYPTOS[self.states.crypto], 
            #     '1m', self.states.ob.start_date, self.states.ob.end_date
            #     ).to_csv("data/1m_df.csv", index=False)
            my_bar.progress(95)
            if self.selected_indicators != []:
                print()
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
                    elif ind == "Stochastic":
                        indicator = StochasticIndicator(**self.indicator_params.get("Stochastic", {}))
                    elif ind == "Stochastic2":
                        indicator = StochasticIndicator(**self.indicator_params.get("Stochastic2", {}), suffix="2")
                    elif ind == "Stochastic3":
                        indicator = StochasticIndicator(**self.indicator_params.get("Stochastic3", {}), suffix="3")
                    elif ind == "Stochastic4":
                        indicator = StochasticIndicator(**self.indicator_params.get("Stochastic4", {}), suffix="4")

                    df = indicator.calculate(df)

            df.to_csv("data/og_backtest.csv", index=False)
            hits = self.states.ob.find_hits(df)
            hits["timestamp"].to_csv("data/filtered_backtest.csv", index=False)
            my_bar.progress(100)
            my_bar.empty()
            st.rerun()

