import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from data_loader import load_and_preprocess

def predict_and_plot():
    # Load data
    x, y, scaler = load_and_preprocess()

    # Load trained model
    model = load_model("../model/lstm_model.h5")

    # Split same as training
    split = int(0.8 * len(x))
    x_test = x[split:]
    y_test = y[split:]

    # Predictions
    predictions = model.predict(x_test)

    # Convert back to original scale
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test)

    # Plot graph
    plt.figure(figsize=(10,5))
    plt.plot(actual, label="Actual Price")
    plt.plot(predictions, label="Predicted Price")
    plt.title("Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    predict_and_plot()