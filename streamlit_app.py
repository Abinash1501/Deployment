import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# Load model and data
model = load_model('model/lstm_model.h5')
df = pd.read_csv('data/Gold_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df[['Price']].dropna()

# Preprocess
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
SEQ_LEN = 60

def forecast_next_30_days():
    input_seq = scaled_data[-SEQ_LEN:]
    forecasted = []

    for _ in range(30):
        inp = input_seq[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
        pred = model.predict(inp)[0]
        forecasted.append(pred)
        input_seq = np.append(input_seq, [pred], axis=0)

    return scaler.inverse_transform(forecasted).flatten()

# Streamlit UI
st.title("📈 30-Day Gold Price Forecast")
st.markdown("Powered by LSTM (Long Short-Term Memory)")

if st.button("Forecast Next 30 Days"):
    forecast = forecast_next_30_days()
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)

    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': forecast})
    st.line_chart(forecast_df.set_index('Date'))

    st.dataframe(forecast_df)
