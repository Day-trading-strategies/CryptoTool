import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
from datetime import datetime

from app.config import *

from app.indicators.half_trend import HalfTrendIndicator
from app.indicators.bollinger_bands import BollingerBandsIndicator
from app.indicators.rsi import RSIIndicator
from app.indicators.williams_r import WilliamsRIndicator
from app.indicators.kdj import KDJIndicator

class OHLCChartCreator:
    """Class to create OHLC charts with optional indicators."""
    
    def __init__(self, selected_cryptos, timeframe, data_fetcher, selected_indicators, indicator_params, states):
        self.df = {crypto: [] for crypto in selected_cryptos}
        self.selected_cryptos = selected_cryptos
        self.selected_indicators = selected_indicators if selected_indicators is not None else []
        self.indicator_params = indicator_params
        self.timeframe = timeframe
        self.data_fetcher = data_fetcher
        self.states = states
        self.render()

    def render(self):
        st.subheader("📈 OHLC Charts")
        if self.states.bt_mode:
            print("bt_mode on")
            crypto = self.states.crypto
            self.states.df = self.data_fetcher.fetch_ohlc_data_range(
                AVAILABLE_CRYPTOS[crypto], 
                self.timeframe, self.states.ob.start_date, self.states.ob.end_date
                )
            self.df[crypto] = self.states.df

            fig = self.create_chart(crypto)
            self.df[crypto].to_csv("backtest_data.csv", index=False)

            # if user chooses a hit point to test, create chart from start to hit point
            if self.states.chart_end:
                temp_df = self.df[crypto].copy()
                self.df[crypto] = self.df[crypto][self.df[crypto]["timestamp"] <= self.states.chart_end]
                fig = self.create_chart(crypto)
                self.df[crypto] = temp_df.copy()
            # else create chart with original data.

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
                hits = self.states.ob.find_hits(self.df[crypto])
                hits["timestamp"].to_csv("filtered_backtest.csv", index=False) 

                timestamps = hits["timestamp"].tolist()
                
                if self.states.chart_end in timestamps:
                    self.states.hit_index = timestamps.index(self.states.chart_end)
                else:
                    self.states.hit_index = 0

                # render a “Next →” button
                if timestamps and st.button("Next →", key="next_hit"):
                    next_idx = (self.states.hit_index + 1) % len(timestamps)
                    self.states.hit_index = next_idx
                    self.states.chart_end = timestamps[next_idx]
                    st.rerun()
                        
                try:
                    filtered = pd.read_csv("filtered_backtest.csv")
                    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"])
                    # turn timestamps into strings for display
                    times = filtered["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
                except FileNotFoundError:
                    times = []

                # Show them in an expander as clickable buttons
                with st.expander(f"Found {len(timestamps)} hit points."):
                    if not times:
                        st.write("No hits found (or filtered_backtest.csv not present).")
                    else:
                        for idx, ts in enumerate(times):
                            # each timestamp is a button; clicking does nothing for now
                            if st.button(ts, key=f"hit_btn_{idx}"):
                                # parse back into a datetime
                                self.states.chart_end = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                                st.rerun()

            else:
                st.error(f"Unable to Backtest Chart")
        # Create tabs for each cryptocurrency
        elif len(self.selected_cryptos) > 1:
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