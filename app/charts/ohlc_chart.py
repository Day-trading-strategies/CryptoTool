import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Optional
from config import *

class OHLCChartCreator:
    """Class to create OHLC charts with optional indicators."""
    
    def __init__(self, df: pd.DataFrame, symbol, timeframe, indicators: Optional[List] = None):
        self.df = df
        self.indicators = indicators if indicators is not None else []
        self.symbol = symbol
        self.timeframe = timeframe
    
    def create_chart(self) -> go.Figure:
        """Create OHLC candlestick chart"""
        if self.df is None or self.df.empty:
            return None
        
        sep_inds = []
        for ind in self.indicators:
            if ind in self.separate_ax_indicators:
                sep_inds.append(ind)
        
        # Changes height of chart as more indicators are added.
        n_rows = 1 + len(sep_inds)
        weights = [3] + [1] * len(sep_inds)
        total_weight = sum(weights)
        row_heights = [w/total_weight for w in weights]

        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes = True,
            vertical_spacing=0.02,
            row_heights = row_heights,
            subplot_titles=(f'{self.symbol} Price Chart ({self.timeframe})',)
        )
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.df['timestamp'],
                open=self.df['open'],
                high=self.df['high'],
                low=self.df['low'],
                close=self.df['close'],
                name='Price',
                increasing_line_color=BULLISH_COLOR,
                decreasing_line_color=BEARISH_COLOR,
                increasing_fillcolor=BULLISH_COLOR,
                decreasing_fillcolor=BEARISH_COLOR
            ),
            row=1, col=1
        )
        fig.update_yaxes(title_text='Price (USDT)', row=1, col=1)

        return fig