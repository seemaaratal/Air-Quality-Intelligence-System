import os
import time
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import load_model

from src.live_air_quality import calculate_aqi
from src.city_search import search_city


MODEL_PATH = "models/multi_city_lstm.keras"
FEATURE_SCALER_PATH = "models/multi_feature_scaler.pkl"
AUX_SCALER_PATH = "models/multi_aux_scaler.pkl"
TARGET_SCALER_PATH = "models/multi_target_scaler.pkl"

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


SEQUENCE_FEATURES = [
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

WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m"
]


@lru_cache(maxsize=1)
def load_artifacts():

    files = [
        MODEL_PATH,
        FEATURE_SCALER_PATH,
        AUX_SCALER_PATH,
        TARGET_SCALER_PATH
    ]

    for path in files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    feature_scaler = joblib.load(
        FEATURE_SCALER_PATH
    )

    aux_scaler = joblib.load(
        AUX_SCALER_PATH
    )

    target_scaler = joblib.load(
        TARGET_SCALER_PATH
    )

    return (
        model,
        feature_scaler,
        aux_scaler,
        target_scaler
    )


def request_json(url, params, attempts=3):

    last_error = None

    for attempt in range(1, attempts + 1):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=(20, 90)
            )

            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as error:

            last_error = error

            if attempt < attempts:
                print(
                    f"API attempt {attempt} failed. Retrying..."
                )
                time.sleep(3)

    raise last_error


def fetch_air_quality(latitude, longitude):

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "pm10,"
            "pm2_5,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone"
        ),
        "past_days": 5,
        "forecast_days": 1,
        "timezone": "Asia/Kolkata"
    }

    data = request_json(
        AIR_URL,
        params
    )

    df = pd.DataFrame(
        data["hourly"]
    )

    df["time"] = pd.to_datetime(
        df["time"]
    )

    return df


def fetch_weather(latitude, longitude):

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m"
        ),
        "past_days": 5,
        "forecast_days": 3,
        "timezone": "Asia/Kolkata"
    }

    data = request_json(
        WEATHER_URL,
        params
    )

    df = pd.DataFrame(
        data["hourly"]
    )

    df["time"] = pd.to_datetime(
        df["time"]
    )

    return df


def add_aqi(df):

    df = df.copy()

    df["pm25_24h"] = (
        df["pm2_5"]
        .rolling(24, min_periods=24)
        .mean()
    )

    df["pm10_24h"] = (
        df["pm10"]
        .rolling(24, min_periods=24)
        .mean()
    )

    df["no2_24h"] = (
        df["nitrogen_dioxide"]
        .rolling(24, min_periods=24)
        .mean()
    )

    df["so2_24h"] = (
        df["sulphur_dioxide"]
        .rolling(24, min_periods=24)
        .mean()
    )

    df["ozone_8h"] = (
        df["ozone"]
        .rolling(8, min_periods=8)
        .mean()
    )

    df["co_8h"] = (
        df["carbon_monoxide"]
        .rolling(8, min_periods=8)
        .mean()
        / 1000
    )

    aqi_values = []

    for _, row in df.iterrows():

        values = [
            row["pm25_24h"],
            row["pm10_24h"],
            row["no2_24h"],
            row["so2_24h"],
            row["ozone_8h"],
            row["co_8h"]
        ]

        if any(pd.isna(value) for value in values):

            aqi_values.append(
                np.nan
            )

            continue

        aqi, _, _ = calculate_aqi(
            row["pm25_24h"],
            row["pm10_24h"],
            row["no2_24h"],
            row["so2_24h"],
            row["ozone_8h"],
            row["co_8h"]
        )

        aqi_values.append(
            aqi
        )

    df["AQI"] = aqi_values

    return df


def add_time_features(df):

    df = df.copy()

    df["hour"] = df["time"].dt.hour
    df["day_of_year"] = df["time"].dt.dayofyear

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_year"] / 365
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_year"] / 365
    )

    return df


def get_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderately Polluted"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


def get_future_weather(
    weather_df,
    target_time
):

    row = weather_df[
        weather_df["time"] == target_time
    ]

    if row.empty:
        raise ValueError(
            f"Future weather unavailable for {target_time}"
        )

    return (
        row[WEATHER_FEATURES]
        .iloc[0]
        .to_numpy(dtype=float)
    )


def predict_multi_city_aqi(
    latitude,
    longitude,
    reference_time=None
):

    (
        model,
        feature_scaler,
        aux_scaler,
        target_scaler
    ) = load_artifacts()

    air_df = fetch_air_quality(
        latitude,
        longitude
    )

    weather_df = fetch_weather(
        latitude,
        longitude
    )

    df = pd.merge(
        air_df,
        weather_df,
        on="time",
        how="inner"
    )

    df = (
        df
        .sort_values("time")
        .drop_duplicates(subset="time")
        .reset_index(drop=True)
    )

    numeric_columns = [
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = add_aqi(df)
    df = add_time_features(df)

    # ----------------------------------------------
    # SAME TIMESTAMP AS CURRENT AQI
    # ----------------------------------------------

    if reference_time is not None:

        reference_time = (
            pd.to_datetime(reference_time)
            .floor("h")
        )

    else:

        reference_time = (
            pd.Timestamp.now(
                tz="Asia/Kolkata"
            )
            .tz_localize(None)
            .floor("h")
        )

    history = df[
        df["time"] <= reference_time
    ].copy()

    history = history.dropna(
        subset=SEQUENCE_FEATURES
    )

    if len(history) < 72:

        raise ValueError(
            "Not enough valid hourly data for LSTM."
        )

    history = (
        history
        .tail(72)
        .copy()
    )

    gaps = (
        history["time"]
        .diff()
        .dropna()
    )

    if not (
        gaps == pd.Timedelta(hours=1)
    ).all():

        raise ValueError(
            "Recent hourly data contains gaps."
        )

    current_time = (
        history["time"]
        .iloc[-1]
    )

    current_aqi = float(
        history["AQI"]
        .iloc[-1]
    )

    aqi_24_ago = float(
        history["AQI"]
        .iloc[-25]
    )

    aqi_48_ago = float(
        history["AQI"]
        .iloc[-49]
    )

    trend24 = (
        current_aqi
        - aqi_24_ago
    )

    previous_trend24 = (
        aqi_24_ago
        - aqi_48_ago
    )

    target_24_time = (
        current_time
        + pd.Timedelta(hours=24)
    )

    target_48_time = (
        current_time
        + pd.Timedelta(hours=48)
    )

    weather24 = get_future_weather(
        weather_df,
        target_24_time
    )

    weather48 = get_future_weather(
        weather_df,
        target_48_time
    )

    sequence_df = history[
        SEQUENCE_FEATURES
    ].copy()

    scaled_sequence = (
        feature_scaler.transform(
            sequence_df
        )
    )

    X = np.expand_dims(
        scaled_sequence,
        axis=0
    )

    auxiliary = np.concatenate([
        weather24,
        weather48,
        [
            current_aqi,
            aqi_24_ago,
            aqi_48_ago,
            trend24,
            previous_trend24,
            float(latitude),
            float(longitude)
        ]
    ])

    auxiliary = auxiliary.reshape(
        1,
        -1
    )

    scaled_aux = (
        aux_scaler.transform(
            auxiliary
        )
    )

    predicted_scaled_delta = (
        model.predict(
            [
                X,
                scaled_aux
            ],
            verbose=0
        )
    )

    predicted_delta = (
        target_scaler
        .inverse_transform(
            predicted_scaled_delta
        )[0]
    )

    prediction_24 = (
        current_aqi
        + predicted_delta[0]
    )

    prediction_48 = (
        current_aqi
        + predicted_delta[1]
    )

    current_aqi = int(
        round(current_aqi)
    )

    prediction_24 = int(
        round(
            np.clip(
                prediction_24,
                0,
                500
            )
        )
    )

    prediction_48 = int(
        round(
            np.clip(
                prediction_48,
                0,
                500
            )
        )
    )

    return {
        "Current_AQI": current_aqi,
        "Prediction_24h": prediction_24,
        "Category_24h": get_category(
            prediction_24
        ),
        "Prediction_48h": prediction_48,
        "Category_48h": get_category(
            prediction_48
        ),
        "Reference_Time": current_time,
        "Target_24h_Time": target_24_time,
        "Target_48h_Time": target_48_time,
        "Latitude": latitude,
        "Longitude": longitude,
        "Model": "Multi-City LSTM"
    }


if __name__ == "__main__":

    query = input(
        "Enter city name: "
    )

    matches = search_city(
        query,
        count=5
    )

    if not matches:

        print(
            "City not found."
        )

        raise SystemExit

    print(
        "\nMatching locations:\n"
    )

    for index, match in enumerate(
        matches,
        start=1
    ):

        print(
            f"{index}. "
            f"{match['display_name']}"
        )

    choice = int(
        input(
            "\nSelect location number: "
        )
    )

    selected = matches[
        choice - 1
    ]

    print(
        f"\nSelected: "
        f"{selected['display_name']}"
    )

    result = predict_multi_city_aqi(
        selected["latitude"],
        selected["longitude"]
    )

    print(
        "\nMULTI-CITY LSTM FORECAST\n"
    )

    print(
        "Current model AQI:",
        result["Current_AQI"]
    )

    print(
        "24-hour AQI:",
        result["Prediction_24h"],
        "-",
        result["Category_24h"]
    )

    print(
        "48-hour AQI:",
        result["Prediction_48h"],
        "-",
        result["Category_48h"]
    )

    print(
        "Reference time:",
        result["Reference_Time"]
    )

    print(
        "24h target:",
        result["Target_24h_Time"]
    )

    print(
        "48h target:",
        result["Target_48h_Time"]
    )

    print(
        "Model:",
        result["Model"]
    )