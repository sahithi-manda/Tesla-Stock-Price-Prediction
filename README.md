# Tesla Stock Price Prediction 📈

A time-series forecasting project that uses **Long Short-Term Memory (LSTM)** neural networks to predict future Tesla (TSLA) adjusted closing prices.

## ✨ Features

- Historical Tesla stock price analysis
- Time-series forecasting using LSTM
- SimpleRNN vs LSTM comparison
- Data preprocessing using MinMaxScaler
- Model evaluation using RMSE
- Streamlit-based prediction interface
- 1, 5 and 10-step forecasting
- Actual vs predicted price visualization

## 🧠 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib
- Jupyter Notebook / Google Colab

## 📁 Project Structure

```text
Tesla-Stock-Price-Prediction/
│
├── Tesla Stock Price Prediction.ipynb
├── app.py
├── lstm_model.h5
├── scaler.pkl
├── requirements.txt
└── README.md
```

## ⚙️ Model Workflow

1. Load historical Tesla stock data.
2. Extract the adjusted closing price.
3. Handle missing values.
4. Normalize data using MinMaxScaler.
5. Create sequences using a 60-day lookback window.
6. Train SimpleRNN and LSTM models.
7. Evaluate predictions using RMSE.
8. Save the trained LSTM model and scaler.
9. Generate future predictions through Streamlit.

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/sahithi-manda/Tesla-Stock-Price-Prediction.git

cd Tesla-Stock-Price-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

## 📊 Using the Application

1. Upload a Tesla historical stock CSV file.
2. Ensure it contains `Date` and `Adj Close` columns.
3. Select your prediction horizon:
   - 1 step
   - 5 steps
   - 10 steps
4. Click **Predict**.
5. View the predicted prices and visualization.

**Note:** The application requires at least 60 historical observations.

## 📓 Training Notebook

The Jupyter notebook contains:

- Data loading and exploration
- Data preprocessing
- Sequence generation
- SimpleRNN implementation
- LSTM implementation
- Model training
- RMSE evaluation
- Actual vs predicted visualization
- Model saving and export

## 🔮 Future Improvements

- Interactive financial dashboards
- Additional forecasting models
- Improved model evaluation
- Downloadable prediction reports
- Real-time market data integration
- Trading-calendar-aware forecasting

## 👩‍💻 Author

**Sahithi Manda**

GitHub: [@sahithi-manda](https://github.com/sahithi-manda)

Project: [Tesla Stock Price Prediction](https://github.com/sahithi-manda/Tesla-Stock-Price-Prediction)

---

**Disclaimer:** This project is developed for educational and research purposes only. Predictions are not financial advice or guaranteed future stock prices.

⭐ If you find this project useful, consider starring the repository!
