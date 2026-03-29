# 📈 AI Stock Price Predictor — LSTM Deep Learning

A stock market prediction system built with LSTM (Long Short-Term Memory) neural networks.
Predicts NSE/BSE stock prices and generates Buy / Sell / Hold trading signals.

## 🚀 Live Demo
> Deployed on Streamlit Cloud — [add your link here after deployment]

## 📌 Features
- **Live prediction** — fetches real-time NSE data via yfinance
- **Buy / Sell / Hold signals** with confidence meter
- **Model performance tab** — RMSE, MAE, loss curve, model comparison
- **LSTM explainer tab** — how the model works, gate-by-gate

## 🛠️ Tech Stack
| Layer | Tool |
|---|---|
| Data | yfinance (Yahoo Finance API) |
| Preprocessing | Pandas, NumPy, Scikit-learn |
| Model | TensorFlow / Keras (LSTM) |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |

## 📁 Project Structure
```
├── app.py              # Streamlit dashboard (main file)
├── train.py            # Train the LSTM model
├── data_loader.py      # Download and preprocess stock data
├── model.py            # LSTM model architecture
├── predict.py          # Offline prediction script
├── lstm_model.h5       # Saved trained model
└── requirements.txt    # Python dependencies
```

## ⚙️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/stock-lstm
cd stock-lstm

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Retrain the model
python train.py

# 4. Launch the app
streamlit run app.py
```

## 🧠 Model Architecture
```
Input      → 60 days of closing prices
LSTM 1     → 50 units, return_sequences=True
Dropout    → 20%
LSTM 2     → 50 units
Dropout    → 20%
Output     → 1 (next day's predicted price)
```
- Optimizer: Adam
- Loss: Mean Squared Error
- Train/Test split: 80% / 20%

## 📊 Results (TCS.NS)
| Metric | Value |
|---|---|
| RMSE | ~₹80–120 |
| MAE | ~₹60–90 |
| Model accuracy | ~93–95% |

## 🔮 Future Work (Research Paper)
- Attention-LSTM architecture for improved accuracy
- Multi-stock comparison (TCS, Infosys, Wipro, HCL)
- Sentiment analysis from NSE financial news
- Target: Expert Systems with Applications (Elsevier Q1)

## 👥 Team
- PAWAR AKSHATA MOHAN
- YOGITHA J

*Mini-Project Submission — ST. JOSEPH'S COLLLEGE OF ENGINEERING — [2026]*
