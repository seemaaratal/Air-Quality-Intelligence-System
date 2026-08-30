import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error


# Load test data
X_test = np.load("data/model_ready/X_test.npy")
y_test = np.load("data/model_ready/y_test.npy")


# Load model and scalers
model = tf.keras.models.load_model(
    "models/aqi_lstm_model.keras"
)

feature_scaler = joblib.load(
    "models/feature_scaler.pkl"
)

target_scaler = joblib.load(
    "models/target_scaler.pkl"
)


# LSTM predictions
predictions = model.predict(X_test)


# Convert scaled values back to actual AQI
actual = target_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).reshape(y_test.shape)

predicted = target_scaler.inverse_transform(
    predictions.reshape(-1, 1)
).reshape(predictions.shape)


# -----------------------------------------
# BASELINE
# -----------------------------------------

# Last known hour from each 48-hour input
last_hour_scaled = X_test[:, -1, :]

last_hour_original = feature_scaler.inverse_transform(
    last_hour_scaled
)

# AQI is feature number 0
current_aqi = last_hour_original[:, 0]

baseline_24 = current_aqi
baseline_48 = current_aqi


# -----------------------------------------
# METRICS
# -----------------------------------------

lstm_mae_24 = mean_absolute_error(
    actual[:, 0],
    predicted[:, 0]
)

lstm_mae_48 = mean_absolute_error(
    actual[:, 1],
    predicted[:, 1]
)

baseline_mae_24 = mean_absolute_error(
    actual[:, 0],
    baseline_24
)

baseline_mae_48 = mean_absolute_error(
    actual[:, 1],
    baseline_48
)


print("\nMODEL COMPARISON")

print("\n24-hour forecast")
print("LSTM MAE    :", round(lstm_mae_24, 2))
print("Baseline MAE:", round(baseline_mae_24, 2))

print("\n48-hour forecast")
print("LSTM MAE    :", round(lstm_mae_48, 2))
print("Baseline MAE:", round(baseline_mae_48, 2))


# -----------------------------------------
# GRAPHS
# -----------------------------------------

os.makedirs("outputs/plots", exist_ok=True)


# First 200 test samples for clear graph
samples = 200


plt.figure(figsize=(12, 5))

plt.plot(
    actual[:samples, 0],
    label="Actual AQI"
)

plt.plot(
    predicted[:samples, 0],
    label="LSTM Predicted AQI"
)

plt.title("24-Hour AQI Forecast: Actual vs Predicted")
plt.xlabel("Test Samples")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/plots/lstm_24h_prediction.png"
)

plt.close()


plt.figure(figsize=(12, 5))

plt.plot(
    actual[:samples, 1],
    label="Actual AQI"
)

plt.plot(
    predicted[:samples, 1],
    label="LSTM Predicted AQI"
)

plt.title("48-Hour AQI Forecast: Actual vs Predicted")
plt.xlabel("Test Samples")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/plots/lstm_48h_prediction.png"
)

plt.close()


print("\nPrediction graphs saved successfully!")