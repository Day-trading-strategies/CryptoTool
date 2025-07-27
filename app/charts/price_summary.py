import streamlit as st
from typing import List

class PriceSummary:
    """Class to create a summary of current prices for multiple cryptocurrencies."""
    
    def __init__(self, selected_cryptos, timeframe, data_fetcher):
        self.selected_cryptos = selected_cryptos
        self.timeframe = timeframe
        self.data_fetcher = data_fetcher

    def display(self):
        """Display price summary cards"""
        cols = st.columns(len(self.selected_cryptos))
        
        for idx, crypto in enumerate(self.selected_cryptos):
            symbol = self.available_cryptos[crypto]
            current_price, price_change = self.data_fetcher.get_current_price(symbol)
            
            with cols[idx]:
                if current_price is not None:
                    change_color = "🟢" if price_change >= 0 else "🔴"
                    st.metric(
                        label=f"{crypto}/USDT",
                        value=f"${current_price:,.4f}",
                        delta=f"{price_change:+.2f}%"
                    )
                else:
                    st.error(f"Unable to fetch {crypto} data")