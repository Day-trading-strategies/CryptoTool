import pandas as pd
import ta
import plotly.graph_objects as go

from .indicator import Indicator

class RSIIndicator(Indicator):
    """RSI (Relative Strength Index) indicator."""
    
    def __init__(self, window: int = 14):
        super().__init__("RSI")
        self.window = window
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI values."""
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=self.window).rsi()
        return df
    
    def add_traces(self, fig: go.Figure, df: pd.DataFrame, row: int = 1):
        """Add RSI trace to figure."""
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['rsi'],
                xaxis="x",
                mode='lines',
                line=dict(width=1),
                name='RSI (14)',
                hovertemplate='RSI: %{y:.1f}<extra></extra>'
            ),
            row=row, col=1
        )
        fig.update_yaxes(title_text='RSI', row=row, col=1)