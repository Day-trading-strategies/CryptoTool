import pandas as pd
import ta
import plotly.graph_objects as go

from .indicator import Indicator

class KDJIndicator(Indicator):
    """KDJ indicator."""
    
    def __init__(self, window: int = 9, smoothing:  int= 3):
        super().__init__("KDJ")
        self.window = window
        self.smoothing = smoothing
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate KDJ values."""
        # compute RSV_t = 100 * (C - LLV(n)) / (HHV(n) - LLV(n))
        low_n  = df['low'].rolling(window=self.window,  min_periods=1).min()
        high_n = df['high'].rolling(window=self.window, min_periods=1).max()
        rsv    = 100 * (df['close'] - low_n) / (high_n - low_n)

        # 2) Wilder’s smoothing for %K and %D
        k = pd.Series(index=df.index, dtype=float)
        d = pd.Series(index=df.index, dtype=float)

        # seed the first value
        k.iloc[0] = rsv.iloc[0]
        d.iloc[0] = k.iloc[0]

        for i in range(1, len(df)):
            k.iloc[i] = (rsv.iloc[i] + (self.smoothing - 1) * k.iloc[i-1]) / self.smoothing
            d.iloc[i] = (k.iloc[i]   + (self.smoothing - 1) * d.iloc[i-1]) / self.smoothing

        j = 3 * k - 2 * d

        df['%K'] = k
        df['%D'] = d
        df['%J'] = j
        return df
    
    def add_traces(self, fig: go.Figure, df: pd.DataFrame, row: int = 1):
        """Add KDJ traces to figure."""
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["%K"],
                mode="lines",
                line=dict(color="rgba(250,50,150,1)", width=1),
                opacity = 0.6,
                name="K (20)"
            ),
            row=row, col=1
        ) 
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["%D"],
                mode="lines",
                xaxis="x",
                line=dict(color="rgba(50,150,255,1)", width=1),
                name="D",
            ),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["%J"],
                mode="lines",
                line=dict(color="rgba(1,250,255,0.8)", width=1),
                name="J",
            ),
            row=row, col=1
        )
        fig.update_yaxes(title_text='KDJ', row=row, col=1)