import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from streamlit_plotly_events import plotly_events
df = pd.read_csv("data/og_backtest.csv", parse_dates=["timestamp"])

def main():
    fig = make_subplots(
                rows=1, cols=1,
                vertical_spacing=0.02)


    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],

            name='Price',
        ),
        row=1, col=1
    )

    plotly_events(fig, 
        
    )
main()