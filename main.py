import streamlit as st
from app.crypto_monitor import CryptoMonitor

if __name__ == "__main__":
    app = CryptoMonitor()
    app.run()