import streamlit as st
import pandas as pd

from app.config import *

class TradeSimulator:
    """ Simlulates Day Trading."""

    def __init__(self, fig, crypto, df, states):
        self.fig = fig
        self.df = df
        self.trading_info = states.trading_info
        self.trade_type = states.trading_info["trade_type"]
        self.stop_loss = states.trading_info["stop_loss"]
        self.take_profit = states.trading_info["take_profit"]
        self.start_price = states.trading_info["start_price"]
        self.entry_time = states.trading_info["entry_time"]
        self.history = states.trading_info["history"]
        self.trading_fee = states.trading_info["trading_fee"]
        self.cur_high = self.df["high"].iloc[-1]
        self.cur_low = self.df["low"].iloc[-1]
        self.cur_price = self.df["close"].iloc[-1]
        self.crypto = crypto

    def render(self):
        key_suffix = f"_{self.crypto}"

        # padding above interface
        st.markdown("<div style='padding-top:200px'></div>", unsafe_allow_html=True)
        
        self._render_wins_losses()
        # No live trade
        if not self.trading_info["trade_on"]:
            self.trade_type = st.selectbox("Trade Type", ["long", "short"], key=f"trade_type{key_suffix}")

            if self.trade_type == "long":
                self.take_profit = st.number_input("Take Profit", min_value=self.cur_price, step=1.0,key=f"take_profit{key_suffix}")
                self.stop_loss = st.number_input("Stop Loss", max_value=self.cur_price, key=f"stop_loss{key_suffix}")
            if self.trade_type == "short":
                self.take_profit = st.number_input("Take Profit", max_value=self.cur_price, key=f"take_profit{key_suffix}")
                self.stop_loss = st.number_input("Stop Loss", min_value=self.cur_price, key=f"stop_loss{key_suffix}")
            
            self.trading_fee = st.number_input("Trading Fee %", max_value=100.0, min_value=0.0, value=0.0, step=0.1, key=f"trading_fee{key_suffix}")
            self.render_lines(self.take_profit, self.stop_loss)

            if st.button("Make Trade", key=f"start{key_suffix}", use_container_width=True):
                self._start_trade()
                st.rerun()

        # Currently Trading + moved at least 1 candlestick.
        if self.trading_info["trade_on"]:
            self._render_details()
            self.render_lines(self.take_profit, self.stop_loss)
            if self.entry_time is None:
                self.trading_info["trade_on"] = False
                st.rerun()
            if self.entry_time < self.df["timestamp"].iloc[-1]:
                self._auto_stop()
                if st.button("Stop Trade", key=f"end{key_suffix}", use_container_width=True):
                    self._end_trade()
                    st.rerun()
            
        if st.button("Reset", key=f"reset{key_suffix}"):
            self._reset_history()

    def _start_trade(self):
        self.trading_info["trade_type"] = self.trade_type
        self.trading_info["take_profit"] = self.take_profit
        self.trading_info["stop_loss"] = self.stop_loss
        self.trading_info["trading_fee"] = self.trading_fee
        self.trading_info["start_price"] = self.df["close"].iloc[-1]
        self.trading_info["entry_time"] = pd.to_datetime(self.df["timestamp"].iloc[-1])
        self.trading_info["trade_on"] = True

    def _auto_stop(self):
        # Market Long
        if self.trade_type == "long":
            if self.cur_high >= self.take_profit:
                change = round(((self.take_profit - self.start_price)/self.start_price) * 100, 2)

                print("Price has hit take profit")
                result = "win" if change - self.trading_fee >= 0 else "loss"
                self._save_trade(result, change - self.trading_fee)
                self._reset_details()
                st.rerun()

            if self.cur_low <= self.stop_loss:
                change = round(((self.stop_loss - self.start_price)/self.start_price) * 100, 2)
                print("Price has hit stop loss")
                self._save_trade("loss", change - self.trading_fee)
                self._reset_details()
                st.rerun()
        
        # Market Short
        if self.trade_type == "short":

            if self.cur_low <= self.take_profit:
                # negative is good
                change = round((self.take_profit - self.start_price)/self.start_price * 100, 2)
                print("Price has hit take profit")
                result = "win" if change + self.trading_fee <= 0 else "loss"
                self._save_trade(result, (change + self.trading_fee)*(-1))
                self._reset_details()
                st.rerun()

            if self.cur_high >= self.stop_loss:
                change = round((self.stop_loss - self.start_price)/self.start_price * 100, 2)
                print("Price has hit stop loss")
                self._save_trade("loss", (change + self.trading_fee)*(-1))
                self._reset_details()
              
                st.rerun()

    def _end_trade(self):
        self.trading_info["trade_on"] = False
        change = round((self.cur_price - self.start_price)/self.start_price * 100, 2)
        if self.trade_type == "long":
            if change - self.trading_fee >= 0:
                self._save_trade("win", change - self.trading_fee)
            if change - self.trading_fee < 0:
                self._save_trade("loss", change - self.trading_fee)
        if self.trade_type == "short":
            if change + self.trading_fee >= 0:
                self._save_trade("loss", change + self.trading_fee)
            if change + self.trading_fee <=0:
                self._save_trade("win", change + self.trading_fee)

        self._reset_details()

    def _render_wins_losses(self):
        wins = self.trading_info["wins"]
        losses = self.trading_info["losses"]
    
        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader(f"Wins: {wins}")
        with col2:
            st.subheader(f"Losses: {losses}")
        try:
            win_rate = wins / (wins+losses)

            if win_rate > 0.6:
                st.subheader(f"Win % : {win_rate:.1%} 🤓")
            else:
                st.subheader(f"Win % : {win_rate:.1%} 🤮")

        except ZeroDivisionError:
            st.subheader("Win %: 0%")

    def _render_details(self):
        st.markdown(
            f"""
            **Market:** {self.trade_type}  
            **Start Price:** {self.trading_info['start_price']}  
            **Stop Loss:** {self.stop_loss}  
            **Take Profit:** {self.take_profit}  
            **Current High:** {self.cur_high}  
            **Current Low:** {self.cur_low}
            """
            )
        
    def _save_trade(self, result: str, change: float):
        new_trade = {
            "result": result,
            "change": f"{change}%",
            "entry_time": self.entry_time,
            "start_price": self.start_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            }
        if result == "win":
            self.trading_info["wins"] += 1
        else:
            self.trading_info["losses"] +=1 
            
        hist = self.trading_info["history"]
        hist = pd.concat([hist, pd.DataFrame([new_trade])], ignore_index=True)

        hist["entry_time"] = pd.to_datetime(hist["entry_time"], errors="coerce")  # add utc=True if you use tz
        self.trading_info["history"] = hist
        self.trading_info["history"].to_csv("data/trading_history.csv")

    def _reset_history(self):
        self.trading_info["history"] = pd.DataFrame()
        self.trading_info["wins"] = 0
        self.trading_info["losses"] = 0
        self.trading_info["entry_time"] = None
        st.rerun()

    def _reset_details(self):
        self.trading_info["trade_type"] = None
        self.trading_info["trade_on"] = False
        self.trading_info["start_price"] = None

    def render_lines(self, take_profit, stop_loss):
        if stop_loss is None:
            self.fig.add_hline(y=self.trading_info["stop_loss"], line_width=1, line_dash="dot", line_color="#FF4B4B",
            annotation_text=f"Stop Loss", annotation_position="right",
            row=1, col=1)
        else:
            self.fig.add_hline(y=stop_loss, line_width=1, line_dash="dot", line_color="#FF4B4B",
            annotation_text=f"Stop Loss", annotation_position="right",
            row=1, col=1)

        if take_profit is None:
            self.fig.add_hline(y=self.trading_info["take_profit"], line_width=1, line_dash="dot", line_color="#A4EDFF",
                annotation_text=f"Take Profit", annotation_position="top right",
                row=1, col=1)
        else:
            self.fig.add_hline(y=take_profit, line_width=1, line_dash="dot", line_color="#A4EDFF",
                annotation_text=f"Take Profit", annotation_position="top right",
                row=1, col=1)
        