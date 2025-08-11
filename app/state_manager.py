import streamlit as st
import pandas as pd

from app.config import *

class SessionStateManager:
    """Manages Streamlit st.session_state variables with getters
    and setters
    """
    DEFAULTS = {
        "bt_mode": False,
        "df": None,
        "crypto": None,
        "timeframe": None,
        "hit_index": 0,
        "chart_end": None,
        "ob": None,
        "highlighted_candles": {},
        "chart_navigation": {}, 
        "previous_timeframe": None,
        "previous_highlights": {},
        "trading_info": {
            "trade_on": False,
            "start_price": None,
            "trade_type": None,
            "entry_time": None,
            "take_profit": None,
            "stop_loss": None,
            "trading_fee": 0.0,
            "history": pd.DataFrame(columns=[
                "entry_time",
                "result", 
                "change",
                "start_price",
                "stop_loss",
                "take_profit"
                ]),
            "wins": 0,
            "losses": 0
        }
    }
    def __init__(self):
        self._initialize_defaults()

    def _initialize_defaults(self):
        for key, default in self.DEFAULTS.items():
            if key not in st.session_state:
                st.session_state[key] = default

    def get(self, key):
        """Generic getter for any session_state key."""
        return st.session_state.get(key)

    def set(self, key, value):
        """Generic setter for any session_state key."""
        st.session_state[key] = value

    @property
    def bt_mode(self) -> bool:
        return st.session_state["bt_mode"]

    @bt_mode.setter
    def bt_mode(self, value: bool):
        st.session_state["bt_mode"] = value

    @property
    def timeframe(self) -> bool:
        return st.session_state["timeframe"]
        
    @timeframe.setter
    def timeframe(self, value: bool):
        st.session_state["timeframe"] = value
    
    @property
    def ob(self):
        return st.session_state["ob"]
    
    @ob.setter
    def ob(self, value):
        st.session_state["ob"] = value

    @property
    def df(self):
        return st.session_state["df"]

    @df.setter
    def df(self, value):
        st.session_state["df"] = value

    @property
    def crypto(self) -> str:
        return st.session_state["crypto"]

    @crypto.setter
    def crypto(self, value: str):
        st.session_state["crypto"] = value

    @property
    def hit_index(self) -> int:
        return st.session_state["hit_index"]

    @hit_index.setter
    def hit_index(self, value: int):
        st.session_state["hit_index"] = value

    @property
    def chart_end(self):
        return st.session_state["chart_end"]

    @chart_end.setter
    def chart_end(self, value):
        st.session_state["chart_end"] = value

    @property
    def highlighted_candles(self) -> dict:
        return st.session_state["highlighted_candles"]

    @highlighted_candles.setter
    def highlighted_candles(self, value: dict):
        st.session_state["highlighted_candles"] = value

    @property
    def chart_navigation(self) -> dict:
        return st.session_state["chart_navigation"]

    @chart_navigation.setter
    def chart_navigation(self, value: dict):
        st.session_state["chart_navigation"] = value

    @property
    def previous_timeframe(self) -> dict:
        return st.session_state["previous_timeframe"]
    
    @previous_timeframe.setter
    def previous_timeframe(self, value):
        st.session_state["previous_timeframe"] = value
    
    @property
    def previous_highlights(self) -> dict:
        return st.session_state["previous_highlights"]
    
    @previous_highlights.setter
    def previous_highlights(self, value: dict):
        st.session_state["previous_highlights"] = value

    @property
    def trading_info(self) -> dict:
        return st.session_state["trading_info"]
    
    @trading_info.setter
    def trading_info(self, value: dict):
        st.session_state["trading_info"] = value