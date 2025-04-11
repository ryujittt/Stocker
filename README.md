# Stocker
Stocker is a PyQt5-based desktop forecasting and pattern-matching tool for OHLCV (Open, High, Low, Close, Volume) financial data. It supports real-time data via the Binance exchange (using ccxt) and allows you to compare recent market data with historical patterns using signal processing and machine learning techniques.

# 🚀 Features
Fetch real-time crypto & forex data from Binance via ccxt

Load historic OHLCV data from local .csv files

Match current data with historical data using correlation or Fourier transform

Forecast short-term future prices using historical analogs

Visualize multiple data transformations including:

Color (bullish/bearish candle)

Slope / Second Slope

Volume / Adjusted Volume

Average Price

Interpolation and logarithmic options for analysis


# 🖥️ GUI Overview
The interface includes:

Symbol selector: Choose from crypto and forex pairs

Timeframe and limit controls

Smoothing (moving average window)

Forecast extension size

Feature extraction method

Plotting variable

Controls for:

Fourier transform

Logarithmic view

Interpolation toggle

Tabbed plots: Original, Matched, Forecasted data

Directory input with autocomplete for loading local CSVs

# 📁 Folder Structure
Ensure you have the following folder on your desktop:


~/Desktop/ohlcv_data/


# Requirements

PyQt5
matplotlib
pandas
numpy
ccxt
scikit-learn
scipy


Disclaimer:
This app is intended for research purposes only. The developers do not take any responsibility for any financial losses or damages resulting from the use of this app. Users are advised to make their own independent decisions and conduct thorough research before taking any actions based on the information provided by this app.

Pansys Project. All rights reserved.

Open-source code, developed by [CHAKRAR ABDELMALIK].
