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
        self.last_adjust_time = states.trading_info["last_adjust_time"]
        self.history = states.trading_info["history"]
        self.trading_fee = states.trading_info["trading_fee"]
        self.cur_high = self.df["high"].iloc[-1]
        self.cur_low = self.df["low"].iloc[-1]
        self.cur_price = self.df["close"].iloc[-1]
        self.crypto = crypto

    def render(self):
        key_suffix = f"_{self.crypto}"

        # padding above interface
        
        # No live trade
        if not self.trading_info["trade_on"]:
            self._render_wins_losses()

            self.trade_type = st.selectbox("Trade Type", ["long", "short"], key=f"trade_type{key_suffix}")

            if self.trade_type == "long":
                self.take_profit = st.number_input("Take Profit", min_value=self.cur_price, step=0.1,key=f"take_profit{key_suffix}")
                self.stop_loss = st.number_input("Stop Loss", value=self.cur_low, key=f"stop_loss{key_suffix}")
            if self.trade_type == "short":
                self.take_profit = st.number_input("Take Profit", max_value=self.cur_price, key=f"take_profit{key_suffix}")
                self.stop_loss = st.number_input("Stop Loss", value=self.cur_high, key=f"stop_loss{key_suffix}")
            
            self.trading_fee = st.number_input("Trading Fee %", max_value=100.0, min_value=0.0, value=0.0, step=0.1, key=f"trading_fee{key_suffix}")

            self.render_lines(self.take_profit, self.stop_loss)

            if st.button("Make Trade", key=f"start{key_suffix}", use_container_width=True):
                self._start_trade()
                st.rerun()

        # Currently Trading + moved at least 1 candlestick.
        if self.trading_info["trade_on"]:
            self._render_details()
            if self.entry_time is None:
                print("entry time was None")
                self.trading_info["trade_on"] = False
                st.rerun()
            if self.trade_type == "long":
                new_profit = st.number_input("Take Profit", value=self.take_profit, step=1.0,key=f"take_profit{key_suffix}")
                new_stop = st.number_input("Stop Loss", value=self.stop_loss, key=f"stop_loss{key_suffix}")
            if self.trade_type == "short":
                new_profit = st.number_input("Take Profit", value=self.take_profit, key=f"take_profit{key_suffix}")
                new_stop = st.number_input("Stop Loss", value=self.stop_loss, key=f"stop_loss{key_suffix}")

            # when SL/TP is edited
            if (new_stop != self.stop_loss) or (new_profit != self.take_profit):
                now_ts = pd.to_datetime(self.df["timestamp"].iloc[-1])
                self.last_adjust_time = now_ts
                self.trading_info["last_adjust_time"] = now_ts

            self.render_lines(new_profit, new_stop)
            self.stop_loss, self.take_profit = new_stop, new_profit
            self.trading_info["stop_loss"], self.trading_info["take_profit"] = new_stop, new_profit

            # call autostop only after a new candle post-entry or post-adjust
            current_ts = pd.to_datetime(self.df["timestamp"].iloc[-1])
            decision_ts_candidates = [t for t in [self.entry_time, self.last_adjust_time] if t is not None]
            last_decision_ts = max(decision_ts_candidates) if decision_ts_candidates else None
            if last_decision_ts is not None and current_ts > last_decision_ts:
                self._auto_stop()

            if st.button("Stop Trade", key=f"end{key_suffix}", use_container_width=True):
                self._end_trade()
                st.rerun()


        if st.button("Reset Trade History", key=f"reset{key_suffix}"):
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
            if self.take_profit is not None and self.cur_high >= self.take_profit:
                gain = round(((self.take_profit - self.start_price)/self.start_price) * 100, 2)
                result = "win" if gain - self.trading_fee >= 0 else "loss"
                print("Price has hit take profit")
                self._save_trade(result, gain - self.trading_fee)
                self._reset_details()
                st.rerun()

            if self.cur_low <= self.stop_loss:
                gain = round(((self.stop_loss - self.start_price)/self.start_price) * 100, 2)
                result = "win" if gain - self.trading_fee >= 0 else "loss"
                print("Price has hit stop loss")
                self._save_trade(result, gain - self.trading_fee)
                self._reset_details()
                st.rerun()
        
        # Market Short
        if self.trade_type == "short":

            if self.take_profit is not None and self.cur_low <= self.take_profit:
                # negative is good
                gain = round((self.take_profit - self.start_price)/self.start_price * 100, 2)
                print("Price has hit take profit")
                result = "win" if gain + self.trading_fee <= 0 else "loss"
                self._save_trade(result, (gain + self.trading_fee)*(-1))
                self._reset_details()
                st.rerun()

            if self.cur_high >= self.stop_loss:
                gain = round((self.stop_loss - self.start_price)/self.start_price * 100, 2)
                print("Price has hit stop loss")
                result = "win" if gain + self.trading_fee <= 0 else "loss"
                self._save_trade(result, (gain + self.trading_fee)*(-1))
                self._reset_details()
              
                st.rerun()

    def _end_trade(self):
        self.trading_info["trade_on"] = False
        gain = round((self.cur_price - self.start_price)/self.start_price * 100, 2)
        if self.trade_type == "long":
            if gain - self.trading_fee >= 0:
                self._save_trade("win", gain - self.trading_fee)
            if gain - self.trading_fee < 0:
                self._save_trade("loss", gain - self.trading_fee)
        if self.trade_type == "short":
            if gain + self.trading_fee >= 0:
                self._save_trade("loss", (gain + self.trading_fee)*(-1))
            if gain + self.trading_fee <=0:
                self._save_trade("win", abs(gain + self.trading_fee))

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

            if win_rate > 0.75:
                st.subheader(f"Win % : {win_rate:.1%} 🔥")
            elif win_rate > 0.5:
                st.subheader(f"Win % : {win_rate:.1%} 🤓")
            else:
                st.subheader(f"Win % : {win_rate:.1%} 🤮")

        except ZeroDivisionError:
            st.subheader("Win %: 0%")

    def _render_details(self):
        st.markdown(
            f"""
            **Market:** {self.trade_type}  
            **Buy Price:** {self.trading_info['start_price']}  
            **Entry Time:** {self.trading_info['entry_time']}
            """
            )
        
    def _save_trade(self, result: str, gain: float):
        new_trade = {
            "result": result,
            "gain": f"{gain}%",
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
        # (Optional) avoid stacking duplicate lines on reruns
        if "shapes" in self.fig.layout and self.fig.layout.shapes:
            self.fig.update_layout(shapes=[])  # clear previously drawn hlines

        # Resolve the y values we should use
        y_stop = stop_loss if stop_loss is not None else self.trading_info.get("stop_loss")
        y_tp   = take_profit if take_profit is not None else self.trading_info.get("take_profit")

        if y_stop is not None:
            self.fig.add_hline(
                y=y_stop, line_width=1, line_dash="dot", line_color="#FF4B4B",
                annotation_text="Stop Loss", annotation_position="right",
                row=1, col=1
            )

        if y_tp is not None:
            self.fig.add_hline(
                y=y_tp, line_width=1, line_dash="dot", line_color="#A4EDFF",
                annotation_text="Take Profit", annotation_position="right",
                row=1, col=1
            )

        