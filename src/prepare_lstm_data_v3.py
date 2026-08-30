import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import MinMaxScaler, StandardScaler


df = pd.read_csv("data/processed/final_aqi_dataset.csv")
df["time"] = pd.to_datetime(df["time"])


# Time features
df["hour"] = df["time"].dt.hour
df["day_of_year"] = df["time"].dt.dayofyear

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)


sequence_features = [
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


weather_features = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m"
]


split_index = int(len(df) * 0.80)

lookback = 72
forecast_24 = 24
forecast_48 = 48


# Scale historical sequence using TRAINING data only
feature_scaler = MinMaxScaler()

feature_scaler.fit(
    df[sequence_features].iloc[:split_index]
)

scaled_sequence = feature_scaler.transform(
    df[sequence_features]
)


X_train = []
aux_train = []
y_train_raw = []

X_test = []
aux_test = []
y_test_raw = []

current_test = []


for i in range(lookback, len(df) - forecast_48 + 1):

    future24_index = i + forecast_24 - 1
    future48_index = i + forecast_48 - 1

    # Past 72 hours
    sequence = scaled_sequence[i - lookback:i]

    current_aqi = df["AQI"].iloc[i - 1]

    aqi_24_ago = df["AQI"].iloc[i - 25]
    aqi_48_ago = df["AQI"].iloc[i - 49]

    future24_aqi = df["AQI"].iloc[future24_index]
    future48_aqi = df["AQI"].iloc[future48_index]

    # Future weather information
    weather24 = df[weather_features].iloc[
        future24_index
    ].to_numpy(dtype=float)

    weather48 = df[weather_features].iloc[
        future48_index
    ].to_numpy(dtype=float)

    # Recent AQI behaviour
    trend24 = current_aqi - aqi_24_ago
    previous_trend24 = aqi_24_ago - aqi_48_ago

    auxiliary = np.concatenate([
        weather24,
        weather48,
        [
            current_aqi,
            aqi_24_ago,
            aqi_48_ago,
            trend24,
            previous_trend24
        ]
    ])

    # Residual / change target
    target = [
        future24_aqi - current_aqi,
        future48_aqi - current_aqi
    ]

    if future48_index < split_index:

        X_train.append(sequence)
        aux_train.append(auxiliary)
        y_train_raw.append(target)

    elif i >= split_index:

        X_test.append(sequence)
        aux_test.append(auxiliary)
        y_test_raw.append(target)
        current_test.append(current_aqi)


X_train = np.array(X_train)
X_test = np.array(X_test)

aux_train = np.array(aux_train)
aux_test = np.array(aux_test)

y_train_raw = np.array(y_train_raw)
y_test_raw = np.array(y_test_raw)

current_test = np.array(current_test)


# Scale auxiliary values
aux_scaler = MinMaxScaler()

aux_train = aux_scaler.fit_transform(aux_train)
aux_test = aux_scaler.transform(aux_test)


# Delta target is better centred around zero
target_scaler = StandardScaler()

y_train = target_scaler.fit_transform(y_train_raw)
y_test = target_scaler.transform(y_test_raw)


os.makedirs("data/model_ready_v3", exist_ok=True)
os.makedirs("models", exist_ok=True)


np.save("data/model_ready_v3/X_train.npy", X_train)
np.save("data/model_ready_v3/X_test.npy", X_test)

np.save("data/model_ready_v3/aux_train.npy", aux_train)
np.save("data/model_ready_v3/aux_test.npy", aux_test)

np.save("data/model_ready_v3/y_train.npy", y_train)
np.save("data/model_ready_v3/y_test.npy", y_test)

np.save(
    "data/model_ready_v3/current_test.npy",
    current_test
)


joblib.dump(
    feature_scaler,
    "models/feature_scaler_v3.pkl"
)

joblib.dump(
    aux_scaler,
    "models/aux_scaler_v3.pkl"
)

joblib.dump(
    target_scaler,
    "models/target_scaler_v3.pkl"
)


print("\nV3 data prepared successfully!")

print("X_train:", X_train.shape)
print("Aux train:", aux_train.shape)
print("y_train:", y_train.shape)

print("\nX_test:", X_test.shape)
print("Aux test:", aux_test.shape)
print("y_test:", y_test.shape)