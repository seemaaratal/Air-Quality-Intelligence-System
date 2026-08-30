import numpy as np
import joblib
import os
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# REPRODUCIBLE RESULTS
# --------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)


# --------------------------------------------------
# LOAD PREPARED DATA
# --------------------------------------------------

X_train = np.load("data/model_ready/X_train.npy")
y_train = np.load("data/model_ready/y_train.npy")

X_test = np.load("data/model_ready/X_test.npy")
y_test = np.load("data/model_ready/y_test.npy")


print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------------
# BUILD LSTM MODEL
# --------------------------------------------------

model = Sequential([
    Input(shape=(48, 11)),

    LSTM(
        64,
        return_sequences=True
    ),

    Dropout(0.2),

    LSTM(32),

    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(2)
])


# --------------------------------------------------
# COMPILE MODEL
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


model.summary()


# --------------------------------------------------
# EARLY STOPPING
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

history = model.fit(
    X_train,
    y_train,

    epochs=40,
    batch_size=32,

    validation_split=0.2,

    callbacks=[early_stopping],

    verbose=1
)


# --------------------------------------------------
# TEST PREDICTIONS
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# CONVERT SCALED AQI BACK TO ORIGINAL AQI
# --------------------------------------------------

target_scaler = joblib.load(
    "models/target_scaler.pkl"
)


actual_aqi = target_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).reshape(y_test.shape)


predicted_aqi = target_scaler.inverse_transform(
    predictions.reshape(-1, 1)
).reshape(predictions.shape)


# --------------------------------------------------
# CALCULATE ERROR
# --------------------------------------------------

mae_24 = mean_absolute_error(
    actual_aqi[:, 0],
    predicted_aqi[:, 0]
)

mae_48 = mean_absolute_error(
    actual_aqi[:, 1],
    predicted_aqi[:, 1]
)

rmse_24 = np.sqrt(
    mean_squared_error(
        actual_aqi[:, 0],
        predicted_aqi[:, 0]
    )
)

rmse_48 = np.sqrt(
    mean_squared_error(
        actual_aqi[:, 1],
        predicted_aqi[:, 1]
    )
)


print("\nMODEL PERFORMANCE")

print("\n24-hour forecast:")
print("MAE :", round(mae_24, 2))
print("RMSE:", round(rmse_24, 2))

print("\n48-hour forecast:")
print("MAE :", round(mae_48, 2))
print("RMSE:", round(rmse_48, 2))


# --------------------------------------------------
# SAMPLE PREDICTIONS
# --------------------------------------------------

print("\nSample predictions:")

for i in range(5):

    print(
        f"\nActual 24h: {actual_aqi[i, 0]:.1f}"
        f" | Predicted 24h: {predicted_aqi[i, 0]:.1f}"
    )

    print(
        f"Actual 48h: {actual_aqi[i, 1]:.1f}"
        f" | Predicted 48h: {predicted_aqi[i, 1]:.1f}"
    )


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    "models/aqi_lstm_model.keras"
)

print("\nLSTM model saved successfully!")