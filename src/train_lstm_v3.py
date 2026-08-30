import numpy as np
import joblib
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Concatenate
)

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# --------------------------------------------------
# RANDOM SEED
# --------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

X_train = np.load(
    "data/model_ready_v3/X_train.npy"
)

X_test = np.load(
    "data/model_ready_v3/X_test.npy"
)

aux_train = np.load(
    "data/model_ready_v3/aux_train.npy"
)

aux_test = np.load(
    "data/model_ready_v3/aux_test.npy"
)

y_train = np.load(
    "data/model_ready_v3/y_train.npy"
)

y_test = np.load(
    "data/model_ready_v3/y_test.npy"
)

current_test = np.load(
    "data/model_ready_v3/current_test.npy"
)


target_scaler = joblib.load(
    "models/target_scaler_v3.pkl"
)


print("Historical input:", X_train.shape)
print("Auxiliary input :", aux_train.shape)


# --------------------------------------------------
# INPUT 1 - HISTORICAL TIME SERIES
# --------------------------------------------------

history_input = Input(
    shape=(72, 15),
    name="history_input"
)

x = LSTM(
    64,
    return_sequences=True
)(history_input)

x = Dropout(0.2)(x)

x = LSTM(32)(x)

x = Dropout(0.2)(x)


# --------------------------------------------------
# INPUT 2 - FUTURE WEATHER + AQI TREND
# --------------------------------------------------

aux_input = Input(
    shape=(13,),
    name="aux_input"
)

a = Dense(
    32,
    activation="relu"
)(aux_input)

a = Dense(
    16,
    activation="relu"
)(a)


# --------------------------------------------------
# COMBINE BOTH INPUTS
# --------------------------------------------------

combined = Concatenate()(
    [x, a]
)

combined = Dense(
    32,
    activation="relu"
)(combined)

combined = Dropout(0.2)(combined)

combined = Dense(
    16,
    activation="relu"
)(combined)


# Two outputs:
# 24-hour AQI change
# 48-hour AQI change

output = Dense(2)(combined)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = Model(
    inputs=[
        history_input,
        aux_input
    ],
    outputs=output
)


model.compile(
    optimizer="adam",
    loss="huber",
    metrics=["mae"]
)


model.summary()


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
    [X_train, aux_train],
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
# PREDICTION
# --------------------------------------------------

predicted_scaled = model.predict(
    [X_test, aux_test]
)


actual_delta = target_scaler.inverse_transform(
    y_test
)

predicted_delta = target_scaler.inverse_transform(
    predicted_scaled
)


# --------------------------------------------------
# FUTURE AQI
# --------------------------------------------------

actual_24 = (
    current_test
    + actual_delta[:, 0]
)

actual_48 = (
    current_test
    + actual_delta[:, 1]
)

predicted_24 = (
    current_test
    + predicted_delta[:, 0]
)

predicted_48 = (
    current_test
    + predicted_delta[:, 1]
)


# Baseline prediction:
# Future AQI = Current AQI

baseline_24 = current_test
baseline_48 = current_test


# --------------------------------------------------
# METRICS
# --------------------------------------------------

lstm_mae_24 = mean_absolute_error(
    actual_24,
    predicted_24
)

baseline_mae_24 = mean_absolute_error(
    actual_24,
    baseline_24
)

lstm_mae_48 = mean_absolute_error(
    actual_48,
    predicted_48
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


print("\nV3 MODEL PERFORMANCE")

print("\n24-hour forecast:")
print(
    "LSTM MAE    :",
    round(lstm_mae_24, 2)
)

print(
    "Baseline MAE:",
    round(baseline_mae_24, 2)
)

print(
    "LSTM RMSE   :",
    round(rmse_24, 2)
)


print("\n48-hour forecast:")

print(
    "LSTM MAE    :",
    round(lstm_mae_48, 2)
)

print(
    "Baseline MAE:",
    round(baseline_mae_48, 2)
)

print(
    "LSTM RMSE   :",
    round(rmse_48, 2)
)


# --------------------------------------------------
# GRAPHS
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
    label="LSTM V3"
)

plt.title(
    "V3 24-Hour AQI Forecast"
)

plt.xlabel("Test Samples")
plt.ylabel("AQI")

plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/plots/lstm_v3_24h.png"
)

plt.close()


plt.figure(figsize=(12, 5))

plt.plot(
    actual_48[:samples],
    label="Actual AQI"
)

plt.plot(
    predicted_48[:samples],
    label="LSTM V3"
)

plt.title(
    "V3 48-Hour AQI Forecast"
)

plt.xlabel("Test Samples")
plt.ylabel("AQI")

plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/plots/lstm_v3_48h.png"
)

plt.close()


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

model.save(
    "models/aqi_lstm_model_v3.keras"
)

print("\nV3 model saved successfully!")