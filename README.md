# Tesla Stock Price Prediction

Forecasts Tesla's adjusted closing price from the previous 60 trading days, comparing a SimpleRNN with an LSTM, and serves the LSTM through a Streamlit app.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

## Problem

Stock prices are a noisy, non-stationary time series. The project asks a narrow question: given the last 60 days of prices, how well can a recurrent network predict the next day? And does an LSTM's extra memory machinery help over a plain RNN on this data?

## Data

Daily TSLA prices in Yahoo Finance CSV format (`Date, Open, High, Low, Close, Adj Close, Volume`), starting 29 June 2010. Only `Adj Close` is used. The CSV is not included in this repository: download TSLA history from Yahoo Finance and save it as `TSLA.csv` next to the notebook.

## Method

```
TSLA.csv → Adj Close → forward-fill gaps → MinMaxScaler (0–1)
   → chronological split: first 80% train, last 20% test
   → sliding windows: 60 days in → next day out
   → SimpleRNN(64) + Dropout(0.2) + Dense(1)
     LSTM(64)      + Dropout(0.2) + Dense(1)
       (Adam, MSE loss, 20% validation split, early stopping, patience 5)
   → inverse-scale predictions → RMSE in USD on the test period
```

## Results

From the saved notebook output (single run):

| Model | Test RMSE (USD) |
|---|---|
| **SimpleRNN** | **13.00** |
| LSTM | 18.79 |

**The simpler model won.** With one input feature and a 60-day window, the dependencies the LSTM's gates are designed for don't add much, and its extra parameters didn't pay off in this run. Neural network training is stochastic, so a fair comparison would average several runs.

The app currently ships the **LSTM** (`lstm_model.h5`), the only model file exported to this repo. Switching to the SimpleRNN is listed under next steps.

## App

```bash
streamlit run app.py
```

Upload a CSV with `Date` and `Adj Close` columns (at least 60 rows) and choose a horizon of 1, 5 or 10 days. Multi-day forecasts are **recursive**: each prediction is fed back in as the newest input. Errors compound with every step, so treat 5- and 10-day outputs as illustrative.

## Project structure

```
├── Tesla Stock Price Prediction.ipynb   # Data prep, SimpleRNN vs LSTM, evaluation
├── app.py                               # Streamlit forecasting app
├── lstm_model.h5                        # Trained LSTM (Keras HDF5)
├── scaler.pkl                           # Fitted MinMaxScaler
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/sahithi-manda/Tesla-Stock-Price-Prediction.git
cd Tesla-Stock-Price-Prediction
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

To retrain, open the notebook. It was written for Google Colab: the `files.upload()` and `files.download()` cells can be skipped when running locally with `TSLA.csv` in the same folder.

## Known limitations

- **Scaling leakage:** the scaler is fit on the full series before the train/test split, so test-period min/max values influence training. Fitting on the training portion only would make the evaluation stricter.
- **No naive baseline.** For daily prices, "tomorrow = today" is a strong baseline. RMSE should be compared against it before claiming the models add value.
- One feature (`Adj Close`) only; volume and returns-based features are unused.

## Next steps

- Fit the scaler on training data only, add a persistence baseline, and report mean ± std RMSE over several seeds.
- Export the SimpleRNN as well and let the app use whichever model performs best.
- Predict returns instead of raw prices, which makes the series closer to stationary.

---

*Educational project, not financial advice.*
