import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd

from app.config import *

from app.indicators.half_trend import HalfTrendIndicator
from app.indicators.bollinger_bands import BollingerBandsIndicator
from app.indicators.rsi import RSIIndicator
from app.indicators.williams_r import WilliamsRIndicator
from app.indicators.kdj import KDJIndicator

class OHLCChartCreator:
    """Class to create OHLC charts with optional indicators."""
    
    def __init__(self, selected_cryptos, timeframe, data_fetcher, selected_indicators, indicator_params):
        self.df = {crypto: [] for crypto in selected_cryptos}
        self.selected_cryptos = selected_cryptos
        self.selected_indicators = selected_indicators if selected_indicators is not None else []
        self.indicator_params = indicator_params
        self.timeframe = timeframe
        self.data_fetcher = data_fetcher
        self.render()

    def render(self):
        st.subheader("📈 OHLC Charts")
        # Create tabs for each cryptocurrency
        if len(self.selected_cryptos) > 1:
            tabs = st.tabs(self.selected_cryptos)
            for idx, crypto in enumerate(self.selected_cryptos):
                with tabs[idx]:
                    self.df[crypto] = self.data_fetcher.fetch_ohlc_data(AVAILABLE_CRYPTOS[crypto], self.timeframe)
                    if self.df[crypto] is not None:
                        fig = self.create_chart(crypto)
                        if fig:
                            st.plotly_chart(
                                fig, 
                                use_container_width=True,
                                config={
                                    'displayModeBar': True,
                                    'scrollZoom': True,  # Enable mouse scroll zoom
                                    'doubleClick': 'reset',  # Double-click to reset zoom
                                    'showTips': False,
                                    'displaylogo': False,
                                    'dragmode': 'pan',  # Set pan as default mode
                                    'modeBarButtonsToRemove': [
                                        'downloadPlot',
                                        'toImage',
                                        'lasso2d',
                                        'select2d',
                                        'zoom2d',         # Remove zoom tool
                                        'zoomIn2d',       # Remove zoom in button
                                        'zoomOut2d',      # Remove zoom out button
                                        'autoScale2d'     # Remove auto scale button
                                    ]
                                }
                            )

                            # Display data table
                            with st.expander(f"📋 {crypto} Data Table"):
                                st.dataframe(self.df[crypto].tail(10).iloc[::-1], use_container_width=True)
                    else:
                        st.error(f"Unable to load chart for {crypto}")
        else:
            crypto = self.selected_cryptos[0]
            self.df[crypto] = self.data_fetcher.fetch_ohlc_data(AVAILABLE_CRYPTOS[crypto], self.timeframe)
            if self.df[crypto] is not None:
                fig = self.create_chart(crypto)
                if fig:
                    st.plotly_chart(
                        fig, 
                        use_container_width=True,
                        config={
                            'displayModeBar': True,
                            'modeBarButtonsToAdd': [
                                ['drawrect']
                            ],
                            'scrollZoom': True,  # Enable mouse scroll zoom
                            'doubleClick': 'reset',  # Double-click to reset zoom
                            'showTips': False,
                            'displaylogo': False,
                            'dragmode': 'pan',  # Set pan as default mode
                            'modeBarButtonsToRemove': [
                                'downloadPlot',
                                'toImage',
                                'lasso2d',
                                'select2d',
                                'zoom2d',         # Remove zoom tool
                                'zoomIn2d',       # Remove zoom in button
                                'zoomOut2d',      # Remove zoom out button
                                'autoScale2d'     # Remove auto scale button
                            ]
                            
                        }
                    )
                    
                    # Display data table
                    with st.expander(f"📋 {crypto} Data Table"):
                        st.dataframe(self.df[crypto].tail(10).iloc[::-1], use_container_width=True)
            else:
                st.error(f"Unable to load chart for {crypto}")


    def create_chart(self, crypto) -> go.Figure:
        """Create OHLC candlestick chart"""
        if self.df[crypto] is None or self.df[crypto].empty:
            return None
        
        sep_inds = []
        for ind in self.selected_indicators:
            if ind in SEPARATE_AX_INDICATORS:
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
            subplot_titles=(f'{crypto} Price Chart ({self.timeframe})',)
        )
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.df[crypto]['timestamp'],
                open=self.df[crypto]['open'],
                high=self.df[crypto]['high'],
                low=self.df[crypto]['low'],
                close=self.df[crypto]['close'],
                name='Price',
                increasing_line_color=BULLISH_COLOR,
                decreasing_line_color=BEARISH_COLOR,
                increasing_fillcolor=BULLISH_COLOR,
                decreasing_fillcolor=BEARISH_COLOR
            ),
            row=1, col=1
        )
        fig.update_yaxes(title_text='Price (USDT)', row=1, col=1)

        for ind in self.selected_indicators:
            if ind == "RSI":
                indicator = RSIIndicator(**self.indicator_params.get("RSI", {}))
            elif ind == "Bollinger Band":
                indicator = BollingerBandsIndicator(**self.indicator_params.get("Bollinger Band", {}))
            elif ind == "KDJ":
                indicator = KDJIndicator(**self.indicator_params.get("KDJ", {}))
            elif ind == "Half Trend":
                indicator = HalfTrendIndicator(**self.indicator_params.get("Half Trend", {}))
            elif ind == "William % Range":
                indicator = WilliamsRIndicator(**self.indicator_params.get("William % Range", {}))
            else:
                continue

            self.df[crypto] = indicator.calculate(self.df[crypto])

            indicator.add_traces(fig, self.df[crypto], 1 if ind not in sep_inds else sep_inds.index(ind) + 2)
            
        # Update layout
        fig.update_layout(
            title=f'{AVAILABLE_CRYPTOS[crypto]} - {self.timeframe} Chart',
            xaxis_rangeslider_visible=False,
            height=CHART_HEIGHT,
            showlegend=False,
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(color='white'),
            dragmode='pan'  # Set pan as default drag mode in layout
        )
        
        return fig