import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from data_loader import load_and_preprocess

def build_model(input_shape):
    model = Sequential()

    # First LSTM layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Second LSTM layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(units=1))

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def train_model():
    x, y, scaler = load_and_preprocess()

    # Split data (80% train, 20% test)
    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    print("Training model...")

    model = build_model((x_train.shape[1], 1))

    model.fit(x_train, y_train, epochs=10, batch_size=32)

    # Save model
    model.save("../model/lstm_model.h5")

    print("Model trained and saved!")

    return model, x_test, y_test, scaler


if __name__ == "__main__":
    train_model()