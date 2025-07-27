import pandas as pd
import ta
import plotly.graph_objects as go

from .indicator import Indicator

class BollingerBandsIndicator(Indicator):
    """Bollinger Bands indicator."""
    
    def __init__(self, window: int = 20, window_dev: float = 2.0):
        super().__init__("Bollinger Band")
        self.window = window
        self.window_dev = window_dev
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close=df["close"], 
            window=self.window, 
            window_dev=self.window_dev
        )
        
        # Add to DataFrame
        df = df.copy()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        return df
    
    def add_traces(self, fig: go.Figure, df: pd.DataFrame, row: int = 1):
        """Add Bollinger Bands traces to figure."""
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["bb_middle"],
                mode="lines",
                line=dict(color="rgba(255,0,0,1)", width=1),
                opacity = 0.6,
                name="BB Middle (20)"
            ),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["bb_upper"],
                mode="lines",
                line=dict(color="rgba(51,153,255,0.8)", width=1),
                name="BB Upper (20,+2σ)",
                hovertemplate="Upper: %{y:.2f}<extra></extra>"
            ),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["bb_lower"],
                mode="lines",
                line=dict(color="rgba(51,153,255,0.8)", width=1),
                name="BB Lower (20,−2σ)",
                hovertemplate="Lower: %{y:.2f}<extra></extra>"
            ),
            row=row, col=1
        )