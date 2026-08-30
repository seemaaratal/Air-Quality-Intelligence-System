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
# ADD TIME FEATURES
# --------------------------------------------------

df["hour"] = df["time"].dt.hour
df["day_of_year"] = df["time"].dt.dayofyear

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["day_sin"] = np.sin(
    2 * np.pi * df["day_of_year"] / 365
)

df["day_cos"] = np.cos(
    2 * np.pi * df["day_of_year"] / 365
)


# --------------------------------------------------
# FEATURES
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
    "wind_speed_10m",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos"
]


# --------------------------------------------------
# TRAIN TEST BOUNDARY
# --------------------------------------------------

split_index = int(len(df) * 0.80)

print("Total records:", len(df))
print("Training boundary:", split_index)


# --------------------------------------------------
# FEATURE SCALING
# --------------------------------------------------

feature_scaler = MinMaxScaler()

feature_scaler.fit(
    df[features].iloc[:split_index]
)

scaled_features = feature_scaler.transform(
    df[features]
)


# --------------------------------------------------
# CREATE DELTA TARGETS
# --------------------------------------------------

lookback = 72

forecast_24 = 24
forecast_48 = 48


X_train = []
y_train_raw = []

X_test = []
y_test_raw = []

current_train = []
current_test = []


for i in range(
    lookback,
    len(df) - forecast_48 + 1
):

    # Previous 72 hours
    X = scaled_features[
        i - lookback:i
    ]

    # Current AQI = last known AQI
    current_aqi = df["AQI"].iloc[i - 1]

    future_24 = df["AQI"].iloc[
        i + forecast_24 - 1
    ]

    future_48 = df["AQI"].iloc[
        i + forecast_48 - 1
    ]

    # Predict CHANGE instead of direct AQI
    delta_24 = future_24 - current_aqi
    delta_48 = future_48 - current_aqi

    y = [delta_24, delta_48]


    if i + forecast_48 - 1 < split_index:

        X_train.append(X)
        y_train_raw.append(y)
        current_train.append(current_aqi)

    elif i >= split_index:

        X_test.append(X)
        y_test_raw.append(y)
        current_test.append(current_aqi)


X_train = np.array(X_train)
X_test = np.array(X_test)

y_train_raw = np.array(y_train_raw)
y_test_raw = np.array(y_test_raw)

current_train = np.array(current_train)
current_test = np.array(current_test)


# --------------------------------------------------
# SCALE TARGET DELTAS
# --------------------------------------------------

target_scaler = MinMaxScaler()

target_scaler.fit(y_train_raw)

y_train = target_scaler.transform(
    y_train_raw
)

y_test = target_scaler.transform(
    y_test_raw
)


# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

print("\nVersion 2 data prepared!")

print("\nX_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nX_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nFeatures:", len(features))


# --------------------------------------------------
# SAVE
# --------------------------------------------------

os.makedirs(
    "data/model_ready_v2",
    exist_ok=True
)

os.makedirs(
    "models",
    exist_ok=True
)


np.save(
    "data/model_ready_v2/X_train.npy",
    X_train
)

np.save(
    "data/model_ready_v2/y_train.npy",
    y_train
)

np.save(
    "data/model_ready_v2/X_test.npy",
    X_test
)

np.save(
    "data/model_ready_v2/y_test.npy",
    y_test
)

np.save(
    "data/model_ready_v2/current_train.npy",
    current_train
)

np.save(
    "data/model_ready_v2/current_test.npy",
    current_test
)


joblib.dump(
    feature_scaler,
    "models/feature_scaler_v2.pkl"
)

joblib.dump(
    target_scaler,
    "models/target_scaler_v2.pkl"
)


print("\nVersion 2 files saved successfully!")