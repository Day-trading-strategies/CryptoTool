import pandas as pd
import ta
import plotly.graph_objects as go
import numpy as np

from .indicator import Indicator

class HalfTrendIndicator(Indicator):
    """Half Trend indicator."""
    
    def __init__(self, amplitude=5):
        super().__init__("Half Trend")
        self.amp = amplitude
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Half Trend calculation based on ATR
        high   = df["high"]
        low    = df["low"]
        close  = df["close"]
        
        # rolling stats
        close_ma      = close.rolling(window=self.amp, min_periods=1).mean()
        highest_high  = high.rolling(window=self.amp, min_periods=1).max()
        lowest_low    = low.rolling(window=self.amp, min_periods=1).min()
        
        hl_t = np.empty(len(df), dtype=float)
        hl_t[0] = close.iloc[0]  # init on first bar
        
        # 2) loop exactly as Pine's `switch`:
        for i in range(1, len(df)):
            prev = hl_t[i-1]
            
            if close_ma.iat[i] < prev and highest_high.iat[i] < prev:
                # price is below last trend AND the recent highs never exceeded it
                hl_t[i] = highest_high.iat[i]
            elif close_ma.iat[i] > prev and lowest_low.iat[i] > prev:
                # price is above last trend AND the recent lows never dipped below it
                hl_t[i] = lowest_low.iat[i]
            else:
                # otherwise carry forward
                hl_t[i] = prev

        half_trend = pd.Series(hl_t, index=df.index, name="halftrend")
        df["half_trend"] = half_trend
        return df
    
    def add_traces(self, fig: go.Figure, df: pd.DataFrame, row: int = 1):
        """Add Half Trend trace to figure."""
        df = self.calculate(df)
        ht = df['half_trend'].values

        # build Pine‐style carry‐forward trend flag
        is_up = np.empty(len(ht), dtype=bool)
        is_up[0] = True  # start assuming “up” (or choose False if you prefer)
        for i in range(1, len(ht)):
            # crossover? (just like ta.crossover)
            if ht[i] > ht[i-1] and ht[i-1] <= (ht[i-2] if i > 1 else ht[0]):
                is_up[i] = True
            # crossunder? (just like ta.crossunder)
            elif ht[i] < ht[i-1] and ht[i-1] >= (ht[i-2] if i > 1 else ht[0]):
                is_up[i] = False
            else:
                # carry prior trend color through all other bars
                is_up[i] = is_up[i-1]

        # stick it back on the df so we can .where() easily
        df['ht_up'] = is_up
        up_mask   = df['ht_up']
        down_mask = ~df['ht_up']

        # plot the two coloured segments
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['half_trend'].where(up_mask),
            mode='lines', line=dict(color='blue', width=2),
            name='HalfTrend ↑'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['half_trend'].where(down_mask),
            mode='lines', showlegend=False,
            line=dict(color='red', width=2)
        ), row=1, col=1)

        # calculate flip‐points
        buy  = up_mask & ~up_mask.shift(1, fill_value=False)
        sell = down_mask & ~down_mask.shift(1, fill_value=False)

        # up and down signal markers
        fig.add_trace(go.Scatter(
            x=df['timestamp'][buy], y=df['half_trend'][buy],
            mode='markers', marker=dict(symbol='triangle-up', size=8, color='blue'),
            name='Buy'
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df['timestamp'][sell], y=df['half_trend'][sell],
            mode='markers', marker=dict(symbol='triangle-down', size=8, color='red'),
            name='Sell'
        ), row=1, col=1)

        # connector lines (there was space between down and up half trends)
        buy_idx  = df.index[buy]
        sell_idx = df.index[sell]

        up_x, up_y = [], []
        for i in buy_idx:
            if i > 0:
                up_x += [df['timestamp'].iat[i-1], df['timestamp'].iat[i], None]
                up_y += [ht[i-1], ht[i], None]

        down_x, down_y = [], []
        for i in sell_idx:
            if i > 0:
                down_x += [df['timestamp'].iat[i-1], df['timestamp'].iat[i], None]
                down_y += [ht[i-1], ht[i], None]

        fig.add_trace(go.Scatter(
            x=up_x, y=up_y, mode='lines', showlegend=False,
            line=dict(color='blue', width=2)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=down_x, y=down_y, mode='lines', showlegend=False,
            line=dict(color='red', width=2)
        ), row=1, col=1)