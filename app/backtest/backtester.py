from datetime import datetime, time
import pandas as pd

from api.binance.data_fetcher import BinanceDataFetcher
from app.config import *

class CryptoBacktester:
    def __init__(self, start_date, end_date, conditions:dict, states):
        self.start_date = datetime.combine(start_date, time.min)
        self.end_date   = datetime.combine(end_date, time.max)
        self.indicator_conditions = conditions
        self.states = states
        
    # Will return timestamps of when all the value become true.
    def find_hits(self, df: pd.DataFrame, epsilon: float = 2.0):       
        conds = []

        # RSI Upper and Lower bound conditions
        has_rsi_up   = "RSI_U" in self.indicator_conditions and self.indicator_conditions["RSI_U"] is not None
        has_rsi_down = "RSI_L" in self.indicator_conditions and self.indicator_conditions["RSI_L"] is not None

        if has_rsi_up and has_rsi_down:
            up   = df["rsi"] >= self.indicator_conditions["RSI_U"]
            down = df["rsi"] <= self.indicator_conditions["RSI_L"]
            conds.append(up | down)
        else:
            if has_rsi_up:
                conds.append(df["rsi"] >= self.indicator_conditions["RSI_U"])
            if has_rsi_down:
                conds.append(df["rsi"] <= self.indicator_conditions["RSI_L"])

        # Stochastic Upper and Lower bound conditions
        st_variants = [
            ("stoch_d",  "ST_U",  "ST_L"),
            ("stoch_d2", "ST2_U", "ST2_L"),
            ("stoch_d3", "ST3_U", "ST3_L"),
            ("stoch_d4", "ST4_U", "ST4_L"),
        ]

        for d_col, up_key, dn_key in st_variants:
            if d_col not in df.columns:
                continue
            has_st_up   = up_key in self.indicator_conditions and self.indicator_conditions[up_key] is not None
            has_st_down = dn_key in self.indicator_conditions and self.indicator_conditions[dn_key] is not None
            if not (has_st_up or has_st_down):
                continue

            series = df[d_col]
            if has_st_up and has_st_down:
                up   = series >= self.indicator_conditions[up_key]
                down = series <= self.indicator_conditions[dn_key]
                conds.append(up | down)
            else:
                if has_st_up:
                    conds.append(series >= self.indicator_conditions[up_key])
                if has_st_down:
                    conds.append(series <= self.indicator_conditions[dn_key])
        

        if "ht_buy" in self.indicator_conditions:
            conds.append(df["ht_up"] == True)
        if "ht_sell" in self.indicator_conditions:
            conds.append(df["ht_up"] == False)
        
        # William % Range Conditions
        has_wr_up   = "WR_U" in self.indicator_conditions and self.indicator_conditions["WR_U"] is not None
        has_wr_down = "WR_L" in self.indicator_conditions and self.indicator_conditions["WR_L"] is not None

        if has_wr_up and has_wr_down:
            up   = df["WR"] >= self.indicator_conditions["WR_U"]
            down = df["WR"] <= self.indicator_conditions["WR_L"]
            conds.append(up | down)
        else:
            if has_wr_up:
                conds.append(df["WR"] >= self.indicator_conditions["WR_U"])
            if has_wr_down:
                conds.append(df["WR"] <= self.indicator_conditions["WR_L"])

        # Bollinger Band touches
        if "Bollinger Top" in self.indicator_conditions:
            conds.append(df["high"] >= df["bb_upper"])

        if "Bollinger Bottom" in self.indicator_conditions:
            conds.append(df["low"] <= df["bb_lower"])

        if "Bollinger Either" in self.indicator_conditions:
            conds.append(
                (df["high"] >= df["bb_upper"])
                | (df["low"] <= df["bb_lower"])
            )

        # KDJ intersections
        if "KDJ" in self.indicator_conditions:
            K, D, J = df["%K"], df["%D"], df["%J"]
            kd_up   = (K.shift(1) < D.shift(1)) & (K >= D)
            kd_down = (K.shift(1) > D.shift(1)) & (K <= D)
            kd_cross = kd_up | kd_down

            kj_up   = (K.shift(1) < J.shift(1)) & (K >= J)
            kj_down = (K.shift(1) > J.shift(1)) & (K <= J)
            kj_cross = kj_up | kj_down

            # require both crossings for a true KDJ “intersection”:
            conds.append(kd_cross & kj_cross)
            # epsilon threshold (you can also make this configurable)
            eps = epsilon
            # only True where all three lines lie within eps of each other
            intersect = ((K - D).abs() < eps) & ((K - J).abs() < eps)
            conds.append(intersect | (kd_cross & kj_cross))

        # combine into an “all‐true” mask        
        all_true = pd.Series(True, index=df.index)
        for c in conds:
            all_true &= c
        # 3) flag only the *first* True of each run
        first_hits = all_true & (~all_true.shift(fill_value=False))

        # Return only the timestamps that made it through all tests
        return df.loc[first_hits].reset_index(drop=True)
