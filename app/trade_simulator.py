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

    # ---------- ROI helpers (no storage changes) ----------
    def _gains_series(self) -> pd.Series:
        """Return gains as numeric % (float), parsed from history['gain'] which may be strings like '10%'."""
        hist = self.trading_info.get("history")
        if not isinstance(hist, pd.DataFrame) or hist.empty or "gain" not in hist.columns:
            return pd.Series(dtype=float)

        s = hist["gain"]
        # Convert both numeric and "10%" string forms to float percent
        if pd.api.types.is_numeric_dtype(s):
            vals = pd.to_numeric(s, errors="coerce")
        else:
            vals = pd.to_numeric(s.astype(str).str.replace("%", "", regex=False), errors="coerce")
        return vals.dropna()

    def _compound_roi_percent(self) -> float:
        """Compound ROI% across closed trades, assuming full capital per trade."""
        gains = self._gains_series()
        if gains.empty:
            return 0.0
        roi = (1.0 + gains / 100.0).prod() - 1.0
        return float(round(roi * 100.0, 2))
    
    # custom st.number_input to prevent edge case/hard crash when hitting stoploss or takeprofits.
    def _safe_number_input(self, label, *, value=None, min_value=None, max_value=None, step=None, key=None):
        EPS = 1e-9
        mv = float(min_value) if min_value is not None else None
        xv = float(max_value) + EPS if max_value is not None else None
        v  = float(value) if value is not None else None
        if v is None:
            # choose a sane default inside the interval if we can
            if mv is not None and xv is not None:
                v = min(max(self.cur_price, mv), xv)
            elif mv is not None:
                v = max(self.cur_price, mv)
            elif xv is not None:
                v = min(self.cur_price, xv)
            else:
                v = self.cur_price
        if mv is not None: v = max(v, mv)
        if xv is not None: v = min(v, xv)
        kwargs = dict(label=label, value=float(v), key=key)
        if mv is not None: kwargs["min_value"] = float(mv)
        if xv is not None: kwargs["max_value"] = float(xv)
        if step is not None: kwargs["step"] = float(step)
        return st.number_input(**kwargs)

    def _refresh_market(self):
        self.cur_high  = float(self.df["high"].iloc[-1])
        self.cur_low   = float(self.df["low"].iloc[-1])
        self.cur_price = float(self.df["close"].iloc[-1])

    def render(self):
        key_suffix = f"_{self.crypto}"
        
        # No live trade
        if not self.trading_info["trade_on"]:
            self._render_wins_losses()

            self.trade_type = st.selectbox("Trade Type", ["long", "short"], key=f"trade_type{key_suffix}")

            if self.trade_type == "long":
                self.take_profit = self._safe_number_input("Take Profit",
                    value=self.take_profit, min_value=self.cur_price, step=0.1, key=f"take_profit{key_suffix}")
                self.stop_loss = self._safe_number_input("Stop Loss",
                    value=self.stop_loss or self.cur_low, key=f"stop_loss{key_suffix}")
            if self.trade_type == "short":
                self.take_profit = self._safe_number_input("Take Profit",
                    value=self.take_profit, max_value=self.cur_price, key=f"take_profit{key_suffix}")
                self.stop_loss = self._safe_number_input("Stop Loss",
                    value=self.stop_loss or self.cur_high, key=f"stop_loss{key_suffix}")

            
            self.trading_fee = st.number_input("Trading Fee %", max_value=100.0, min_value=0.0, value=0.0, step=0.1, key=f"trading_fee{key_suffix}")

            self.render_lines(self.take_profit, self.stop_loss)

            if st.button("Make Trade", key=f"start{key_suffix}", use_container_width=True):
                self._start_trade()
                st.rerun()

        # Currently Trading
        if self.trading_info["trade_on"]:
            self._render_details()

            if self.entry_time is None:
                print("entry time was None")
                self.trading_info["trade_on"] = False
                st.rerun()

            # Allowing to Edit SL/TP
            if self.trade_type == "long":
                new_profit = st.number_input("Take Profit", value=self.take_profit, step=1.0, key=f"take_profit{key_suffix}")
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
            # grabs the later time of entry_time or last adjusted time.
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

    def _net_pct(self, exit_price: float) -> float:
        """Signed net % P&L including fees. Positive = profit."""
        raw = (exit_price - self.start_price) / self.start_price * 100.0
        return (raw - self.trading_fee) if self.trade_type == "long" else (-raw - self.trading_fee)

    def _auto_stop(self):
        # Always operate on fresh values
        print('auto stop activated.')
        self._refresh_market()

        tp, sl = self.take_profit, self.stop_loss
        exit_price = None

        if self.trade_type == "long":
            # TP first to catch wick-throughs
            if tp is not None and self.cur_high >= tp:
                exit_price = tp
            elif sl is not None and self.cur_low <= sl:
                exit_price = sl
        elif self.trade_type == "short":  # short
            if tp is not None and self.cur_low <= tp:
                exit_price = tp
            elif sl is not None and self.cur_high >= sl:
                exit_price = sl

        if exit_price is None:
            return  # nothing to do this tick

        net = self._net_pct(exit_price)
        threshold = 0.05  # same "even" band as _end_trade
        if abs(net) < threshold:
            result = "even"  
        elif net > 0:
            result = "win"
        else:
            result = "loss"

        self._save_trade(result, round(net, 2))
        self._reset_details()
        st.rerun()


    def _end_trade(self):
        """ When Shorting, we add trading_fee to the change.
            When Long, we subtract trading_fee to the change"""
        self.trading_info["trade_on"] = False

        raw = (self.cur_price - self.start_price) / self.start_price * 100.0
        # Long:  profit = raw - fee
        # Short: profit = -raw - fee
        net = (raw - self.trading_fee) if self.trade_type == "long" else (-raw - self.trading_fee)

        # classify BEFORE rounding (keeps threshold precise); 0% counts as win (your current behavior)
        threshold = 0.05
        print(net)
        if abs(net) < threshold:
            result = "even"
        elif net > 0:
            result = "win"
        else:
            result = "loss"

        self._save_trade(result, round(net, 2))
        self._reset_details()

    def _render_wins_losses(self):
        wins = self.trading_info["wins"]
        losses = self.trading_info["losses"]
        evens = self.trading_info["evens"]

        roi_pct = self._compound_roi_percent()
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            st.markdown(f"<div style='font-size:{"20px"}'>Wins: {wins}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div style='font-size:{"20px"}'>Losses: {losses}</div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div style='font-size:{"20px"}'>Evens: {evens}</div>", unsafe_allow_html=True)

        # Win rate face
        col1, col2 = st.columns([1,1])
        with col1:
            try:
                win_rate = wins / (wins + losses)
                if win_rate > 0.75:
                    st.markdown(f"<div style='font-size:{"22px"}'>WR : {win_rate:.1%} 🔥</div>", unsafe_allow_html=True)
                elif win_rate > 0.5:
                    st.markdown(f"<div style='font-size:{"22px"}'>WR : {win_rate:.1%} 🤓</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='font-size:{"22px"}'>WR : {win_rate:.1%} 🤮</div>", unsafe_allow_html=True)
            except ZeroDivisionError:
                st.markdown(f"<div style='font-size:{"26px"}'>WR : 0%</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div style='font-size:{"26px"}'>ROI: {roi_pct:.2f}%</div>", unsafe_allow_html=True)

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
            "gain": f"{gain}%",  # keep your existing storage format
            "entry_time": self.entry_time,
            "start_price": self.start_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }
        if result == "win":
            self.trading_info["wins"] += 1
        elif result == "loss":
            self.trading_info["losses"] += 1
        else:
            self.trading_info["evens"] += 1
            
        hist = self.trading_info["history"]
        hist = pd.concat([hist, pd.DataFrame([new_trade])], ignore_index=True)

        hist["entry_time"] = pd.to_datetime(hist["entry_time"], errors="coerce")  # add utc=True if you use tz
        self.trading_info["history"] = hist
        self.trading_info["history"].to_csv("data/trading_history.csv")

    def _reset_history(self):
        self.trading_info["history"] = pd.DataFrame()
        self.trading_info["wins"] = 0
        self.trading_info["losses"] = 0
        self.trading_info["evens"] = 0
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
