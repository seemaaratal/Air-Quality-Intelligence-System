import numpy as np
import joblib
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# RANDOM SEED
# --------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

X_train = np.load("data/model_ready_v2/X_train.npy")
y_train = np.load("data/model_ready_v2/y_train.npy")

X_test = np.load("data/model_ready_v2/X_test.npy")
y_test = np.load("data/model_ready_v2/y_test.npy")

current_test = np.load(
    "data/model_ready_v2/current_test.npy"
)

target_scaler = joblib.load(
    "models/target_scaler_v2.pkl"
)


print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------------
# BUILD MODEL
# --------------------------------------------------

model = Sequential([
    Input(shape=(72, 15)),

    LSTM(
        64,
        return_sequences=True
    ),

    Dropout(0.2),

    LSTM(32),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(2)
])


# --------------------------------------------------
# COMPILE
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="huber",
    metrics=["mae"]
)


# --------------------------------------------------
# CALLBACKS
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    patience=3,
    factor=0.5,
    min_lr=0.00001
)


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

history = model.fit(
    X_train,
    y_train,

    epochs=50,
    batch_size=32,

    validation_split=0.2,

    shuffle=False,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

predicted_scaled = model.predict(X_test)


# Convert scaled delta back to original AQI change
actual_delta = target_scaler.inverse_transform(
    y_test
)

predicted_delta = target_scaler.inverse_transform(
    predicted_scaled
)


# --------------------------------------------------
# CONVERT DELTA INTO FUTURE AQI
# --------------------------------------------------

actual_24 = current_test + actual_delta[:, 0]
actual_48 = current_test + actual_delta[:, 1]

predicted_24 = current_test + predicted_delta[:, 0]
predicted_48 = current_test + predicted_delta[:, 1]


# Baseline:
# future AQI = current AQI
baseline_24 = current_test
baseline_48 = current_test


# --------------------------------------------------
# METRICS
# --------------------------------------------------

lstm_mae_24 = mean_absolute_error(
    actual_24,
    predicted_24
)

lstm_mae_48 = mean_absolute_error(
    actual_48,
    predicted_48
)

baseline_mae_24 = mean_absolute_error(
    actual_24,
    baseline_24
)

baseline_mae_48 = mean_absolute_error(
    actual_48,
    baseline_48
)


rmse_24 = np.sqrt(
    mean_squared_error(
        actual_24,
        predicted_24
    )
)

rmse_48 = np.sqrt(
    mean_squared_error(
        actual_48,
        predicted_48
    )
)


print("\nV2 MODEL PERFORMANCE")

print("\n24-hour forecast:")
print("LSTM MAE    :", round(lstm_mae_24, 2))
print("Baseline MAE:", round(baseline_mae_24, 2))
print("LSTM RMSE   :", round(rmse_24, 2))

print("\n48-hour forecast:")
print("LSTM MAE    :", round(lstm_mae_48, 2))
print("Baseline MAE:", round(baseline_mae_48, 2))
print("LSTM RMSE   :", round(rmse_48, 2))


# --------------------------------------------------
# SAVE GRAPH
# --------------------------------------------------

os.makedirs(
    "outputs/plots",
    exist_ok=True
)

samples = 200

plt.figure(figsize=(12, 5))

plt.plot(
    actual_24[:samples],
    label="Actual AQI"
)

plt.plot(
    predicted_24[:samples],
    label="LSTM V2"
)

plt.title(
    "LSTM V2 - 24 Hour AQI Prediction"
)

plt.xlabel("Test Samples")
plt.ylabel("AQI")

plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/plots/lstm_v2_24h.png"
)

plt.close()


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

model.save(
    "models/aqi_lstm_model_v2.keras"
)

print("\nLSTM V2 model saved successfully!")