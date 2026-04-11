import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("lstm_model.h5", compile=False)

import matplotlib.pyplot as plt

st.set_page_config(page_title="Tesla Stock Predictor")

st.title("Tesla Stock Price Prediction")
st.write("Forecast Tesla closing prices using an LSTM model.")

# Load artifacts
model = load_model("lstm_model.h5", compile=False)
model.compile(optimizer="adam", loss="mse", metrics=["mse"])

scaler = joblib.load("scaler.pkl")

WINDOW = 60   # use the same window you trained with


def predict_future(last_seq, horizon):
    seq = last_seq.copy()
    preds = []

    for _ in range(horizon):
        p = model.predict(seq.reshape(1, seq.shape[0], seq.shape[1]), verbose=0)
        preds.append(p[0][0])
        seq = np.vstack([seq[1:], p])

    preds = np.array(preds).reshape(-1, 1)
    return scaler.inverse_transform(preds)


st.subheader("Upload Tesla CSV (must include Adj Close column)")
file = st.file_uploader("Upload file", type=["csv"])

if file:
    df = pd.read_csv(file)

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index("Date", inplace=True)

    values = df[['Adj Close']].values
    scaled = scaler.transform(values)

    if len(values) < WINDOW:
        st.error(f"You need at least {WINDOW} rows.")
    else:
        last_seq = scaled[-WINDOW:]

        horizon = st.selectbox("Select horizon:", [1, 5, 10])

        if st.button("Predict"):
            preds = predict_future(last_seq, horizon)

            future_dates = pd.date_range(df.index[-1], periods=horizon+1, freq='D')[1:]
            result = pd.DataFrame({"Predicted Close": preds.flatten()}, index=future_dates)

            st.success("Prediction generated:")
            st.dataframe(result)

            plt.figure(figsize=(8,4))
            plt.plot(df.index[-60:], df['Adj Close'].tail(60), label="Actual")
            plt.plot(result.index, result["Predicted Close"], label="Forecast")
            plt.legend()
            st.pyplot(plt)

else:
    st.info("Upload a TSLA CSV file to start prediction.")
