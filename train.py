"""
train.py — Train the LSTM model on TCS.NS data and save it.
Run this once: python train.py
The saved lstm_model.h5 is then used by app.py
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from data_loader import load_and_preprocess

TICKER   = "TCS.NS"
EPOCHS   = 20
BATCH    = 32
SEQ_LEN  = 60

def build_model(seq_len):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model

def train():
    print(f"Downloading data for {TICKER}...")
    X, Y, scaler = load_and_preprocess(TICKER)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    model = build_model(SEQ_LEN)

    # Stop early if validation loss stops improving
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("Training...")
    history = model.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH,
        validation_data=(X_test, Y_test),
        callbacks=[early_stop],
        verbose=1
    )

    model.save("lstm_model.h5")
    print("Model saved as lstm_model.h5")

    # Quick accuracy check
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    preds = scaler.inverse_transform(model.predict(X_test))
    actual = scaler.inverse_transform(Y_test)
    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae = mean_absolute_error(actual, preds)
    print(f"Test RMSE: ₹{rmse:.2f}  |  MAE: ₹{mae:.2f}")

if __name__ == "__main__":
    train()
