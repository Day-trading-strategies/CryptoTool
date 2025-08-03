import pandas as pd
import ta
import plotly.graph_objects as go

from .indicator import Indicator

class WilliamsRIndicator(Indicator):
    """Williams %R indicator."""
    
    def __init__(self, lbp: int = 14):
        super().__init__("William % Range")
        self.lbp = lbp
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Williams %R values."""
        df["WR"] = ta.momentum.williams_r(
            df["high"], df["low"], df["close"], lbp=self.lbp
        )
        return df
    

    def add_traces(self, fig: go.Figure, df: pd.DataFrame, row: int = 1):
        """Add Williams %R trace to figure."""
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['WR'],
                mode='lines',
                line=dict(width=1),
                name='Williams %R (14)',
                hovertemplate='Williams %R: %{y:.1f}<extra></extra>'
            ),
            row=row, col=1
        )
        fig.update_yaxes(title_text='Williams %R', row=row, col=1)

        