import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/processed/final_aqi_dataset.csv")

df["time"] = pd.to_datetime(df["time"])


# --------------------------------------------------
# FEATURES USED BY LSTM
# --------------------------------------------------

features = [
    "AQI",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m"
]


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

split_index = int(len(df) * 0.80)

print("Total records:", len(df))
print("Training boundary:", split_index)


# --------------------------------------------------
# SCALING
# --------------------------------------------------

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Fit scaler ONLY on training data
feature_scaler.fit(
    df[features].iloc[:split_index]
)

target_scaler.fit(
    df[["AQI"]].iloc[:split_index]
)

# Transform complete dataset
scaled_features = feature_scaler.transform(
    df[features]
)

scaled_target = target_scaler.transform(
    df[["AQI"]]
)


# --------------------------------------------------
# CREATE LSTM SEQUENCES
# --------------------------------------------------

lookback = 48

forecast_24 = 24
forecast_48 = 48

X_train = []
y_train = []

X_test = []
y_test = []


for i in range(
    lookback,
    len(df) - forecast_48 + 1
):

    # Previous 48 hours
    X = scaled_features[
        i - lookback:i
    ]

    # AQI 24 hours after prediction point
    y_24 = scaled_target[
        i + forecast_24 - 1
    ][0]

    # AQI 48 hours after prediction point
    y_48 = scaled_target[
        i + forecast_48 - 1
    ][0]

    y = [y_24, y_48]

    # Training data
    if i + forecast_48 - 1 < split_index:

        X_train.append(X)
        y_train.append(y)

    # Testing data
    elif i >= split_index:

        X_test.append(X)
        y_test.append(y)


# Convert lists into NumPy arrays

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)


# --------------------------------------------------
# DISPLAY SHAPES
# --------------------------------------------------

print("\nLSTM data prepared successfully!")

print("\nX_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\nX_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# --------------------------------------------------
# SAVE DATA
# --------------------------------------------------

os.makedirs(
    "data/model_ready",
    exist_ok=True
)

os.makedirs(
    "models",
    exist_ok=True
)


np.save(
    "data/model_ready/X_train.npy",
    X_train
)

np.save(
    "data/model_ready/y_train.npy",
    y_train
)

np.save(
    "data/model_ready/X_test.npy",
    X_test
)

np.save(
    "data/model_ready/y_test.npy",
    y_test
)


# Save scalers for later prediction

joblib.dump(
    feature_scaler,
    "models/feature_scaler.pkl"
)

joblib.dump(
    target_scaler,
    "models/target_scaler.pkl"
)


print("\nPrepared data and scalers saved successfully!")