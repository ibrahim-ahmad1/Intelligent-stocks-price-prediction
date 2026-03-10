import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from modules.hybrid_engine import generate_trade_signal
from modules.risk_analysis import analyze_risk

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Stock Intelligence System", layout="wide")
st.title("📈 AI-Powered Stock Market Intelligence Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
stock_path = "data/processed/stock_with_indicators.csv"
sentiment_path = "data/processed/daily_news_sentiment.csv"
model_path = "models/lstm_model.h5"
scaler_path = "models/scaler.pkl"

df = pd.read_csv(stock_path)

# -----------------------------
# SHOW PRICE CHART
# -----------------------------
st.subheader("📊 Stock Price Chart")
st.line_chart(df["Close"])

# -----------------------------
# LSTM PREDICTION
# -----------------------------
st.subheader("🔮 Next Day Price Prediction")

model = load_model(model_path)
scaler = joblib.load(scaler_path)

close_prices = df[["Close"]].values
scaled_data = scaler.transform(close_prices)

window_size = 60
last_60 = scaled_data[-window_size:]
X_input = np.array([last_60])

predicted_scaled = model.predict(X_input)
predicted_price = scaler.inverse_transform(predicted_scaled)

st.success(f"Predicted Next Closing Price: ₹ {round(float(predicted_price[0][0]), 2)}")

# -----------------------------
# HYBRID DECISION
# -----------------------------
st.subheader("📢 Trading Decision")

decision = generate_trade_signal(stock_path, sentiment_path)

st.write(f"### Signal: {decision['signal']}")
st.write(f"RSI: {decision['rsi']}")
st.write(f"Sentiment Score: {decision['sentiment_score']}")
st.write(f"Trend: {decision['ma_trend']}")

# -----------------------------
# RISK ANALYSIS
# -----------------------------
st.subheader("⚠ Risk Analysis")

risk_info = analyze_risk(stock_path)

st.write(f"Volatility: {risk_info['volatility']}")
st.write(f"Max Drawdown: {risk_info['max_drawdown']}")
st.write(f"Risk Level: {risk_info['risk_level']}")
