import pandas as pd
import ta
import plotly.graph_objects as go

from .indicator import Indicator

class HalfTrendIndicator(Indicator):
    """Half Trend indicator."""
    
    def __init__(self, period: int = 10, multiplier: float = 1.0):
        super().__init__("Half Trend")
        self.period = period
        self.multiplier = multiplier
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Half Trend calculation based on ATR
        hl2 = (df['high'] + df['low']) / 2
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=self.period).average_true_range()
        # Initialize with first True Range
        ht = [hl2.iloc[0]]
        trend = True  # True=uptrend, False=downtrend
        for i in range(1, len(df)):
            delta = atr.iloc[i] * self.multiplier
            prev_ht = ht[-1]
            if hl2.iloc[i] > prev_ht + delta:
                ht.append(prev_ht + delta)
                trend = True
            elif hl2.iloc[i] < prev_ht - delta:
                ht.append(prev_ht - delta)
                trend = False
            else:
                ht.append(prev_ht)
        df['half_trend'] = ht

        return df
    
    def add_traces(self, fig: go.Figure, df: pd.DataFrame, row: int = 1):
        """Add Half Trend trace to figure."""
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['half_trend'],
                mode='lines', 
                name='Half Trend',
                line=dict(width=1)
            ), row=row, col=1
        )