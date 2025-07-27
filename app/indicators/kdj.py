import pandas as pd
import ta
import plotly.graph_objects as go

from .indicator import Indicator

class KDJIndicator(Indicator):
    """KDJ indicator."""
    
    def __init__(self, period: int = 14, signal: int = 3):
        super().__init__("KDJ")
        self.period = period
        self.signal = signal
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate KDJ values."""
        # Calculate KDJ using Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            df["high"], df["low"], df["close"], 
            window=self.period, smooth_window=self.signal
        )
        
        # Add to DataFrame
        df = df.copy()
        df['%K'] = stoch.stoch()
        df['%D'] = stoch.stoch_signal()
        df['%J'] = 3 * df['%K'] - 2 * df['%D']
        return df
    
    def add_traces(self, fig: go.Figure, df: pd.DataFrame, row: int = 1):
        """Add KDJ traces to figure."""
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["%K"],
                mode="lines",
                line=dict(color="rgba(255,0,0,1)", width=1),
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
                line=dict(color="rgba(51,153,255,0.8)", width=1),
                name="D",
            ),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["%J"],
                mode="lines",
                line=dict(color="rgba(51,153,255,0.8)", width=1),
                name="J",
            ),
            row=row, col=1
        )
        fig.update_yaxes(title_text='KDJ', row=row, col=1)