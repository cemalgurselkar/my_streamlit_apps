import streamlit as st
import yfinance as yf
import pandas as pd

st.write("""
#Simple Stock Price App
""")

ticker_symbol = 'GOOG'
tickerData = yf.Ticker(ticker_symbol)
ticker_df = tickerData.history(period='max', start='2010-5-31', end='2020-5-31')

st.line_chart(ticker_df.Close)
st.line_chart(ticker_df.Volume)