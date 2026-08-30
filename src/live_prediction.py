import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import load_model

from src.live_air_quality import calculate_aqi


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

MODEL_PATH = "models/aqi_lstm_model_v3.keras"

FEATURE_SCALER_PATH = "models/feature_scaler_v3.pkl"
AUX_SCALER_PATH = "models/aux_scaler_v3.pkl"
TARGET_SCALER_PATH = "models/target_scaler_v3.pkl"


# --------------------------------------------------
# MODEL FEATURES
# Exact same order as training
# --------------------------------------------------

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


# --------------------------------------------------
# LOAD MODEL + SCALERS
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_artifacts():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
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


# --------------------------------------------------
# FETCH AIR QUALITY
# --------------------------------------------------

def fetch_air_quality(
    latitude,
    longitude
):

    url = (
        "https://air-quality-api.open-meteo.com/"
        "v1/air-quality"
    )

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "pm10,pm2_5,carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,ozone"
        ),
        "past_days": 5,
        "forecast_days": 1,
        "timezone": "Asia/Kolkata"
    }

    response = requests.get(
        url,
        params=params,
        timeout=(20,90)
    )

    response.raise_for_status()

    data = response.json()["hourly"]

    df = pd.DataFrame(data)

    df["time"] = pd.to_datetime(
        df["time"]
    )

    return df


# --------------------------------------------------
# FETCH WEATHER
# Includes future weather needed by V3
# --------------------------------------------------

def fetch_weather(
    latitude,
    longitude
):

    url = (
        "https://api.open-meteo.com/v1/forecast"
    )

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

    response = requests.get(
        url,
        params=params,
        timeout=(20,90)
    )

    response.raise_for_status()

    data = response.json()["hourly"]

    df = pd.DataFrame(data)

    df["time"] = pd.to_datetime(
        df["time"]
    )

    return df


# --------------------------------------------------
# ADD ROLLING CPCB-STYLE AQI
# --------------------------------------------------

def add_aqi(df):

    df = df.copy()

    # 24-hour averages
    df["pm25_avg"] = (
        df["pm2_5"]
        .rolling(
            window=24,
            min_periods=24
        )
        .mean()
    )

    df["pm10_avg"] = (
        df["pm10"]
        .rolling(
            window=24,
            min_periods=24
        )
        .mean()
    )

    df["no2_avg"] = (
        df["nitrogen_dioxide"]
        .rolling(
            window=24,
            min_periods=24
        )
        .mean()
    )

    df["so2_avg"] = (
        df["sulphur_dioxide"]
        .rolling(
            window=24,
            min_periods=24
        )
        .mean()
    )

    # 8-hour averages
    df["ozone_avg"] = (
        df["ozone"]
        .rolling(
            window=8,
            min_periods=8
        )
        .mean()
    )

    # µg/m³ -> mg/m³
    df["co_avg"] = (
        df["carbon_monoxide"]
        .rolling(
            window=8,
            min_periods=8
        )
        .mean()
        / 1000
    )


    aqi_values = []


    for _, row in df.iterrows():

        required = [
            row["pm25_avg"],
            row["pm10_avg"],
            row["no2_avg"],
            row["so2_avg"],
            row["ozone_avg"],
            row["co_avg"]
        ]

        if any(
            pd.isna(value)
            for value in required
        ):

            aqi_values.append(
                np.nan
            )

            continue


        aqi, _, _ = calculate_aqi(
            row["pm25_avg"],
            row["pm10_avg"],
            row["no2_avg"],
            row["so2_avg"],
            row["ozone_avg"],
            row["co_avg"]
        )

        aqi_values.append(
            aqi
        )


    df["AQI"] = aqi_values

    return df


# --------------------------------------------------
# TIME FEATURES
# --------------------------------------------------

def add_time_features(df):

    df = df.copy()

    df["hour"] = (
        df["time"].dt.hour
    )

    df["day_of_year"] = (
        df["time"].dt.dayofyear
    )


    df["hour_sin"] = np.sin(
        2
        * np.pi
        * df["hour"]
        / 24
    )

    df["hour_cos"] = np.cos(
        2
        * np.pi
        * df["hour"]
        / 24
    )


    df["day_sin"] = np.sin(
        2
        * np.pi
        * df["day_of_year"]
        / 365
    )

    df["day_cos"] = np.cos(
        2
        * np.pi
        * df["day_of_year"]
        / 365
    )

    return df


# --------------------------------------------------
# AQI CATEGORY
# --------------------------------------------------

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


# --------------------------------------------------
# GET FUTURE WEATHER ROW
# --------------------------------------------------

def get_future_weather(
    df,
    target_time
):

    row = df[
        df["time"] == target_time
    ]

    if row.empty:

        raise ValueError(
            f"Weather unavailable for {target_time}"
        )

    return (
        row[
            WEATHER_FEATURES
        ]
        .iloc[0]
        .to_numpy(
            dtype=float
        )
    )


# --------------------------------------------------
# LIVE LSTM PREDICTION
# --------------------------------------------------

def predict_aqi(
    latitude=15.8521,
    longitude=74.5045
):

    (
        model,
        feature_scaler,
        aux_scaler,
        target_scaler

    ) = load_artifacts()


    # ----------------------------------------------
    # FETCH DATA
    # ----------------------------------------------

    air_df = fetch_air_quality(
        latitude,
        longitude
    )

    weather_df = fetch_weather(
        latitude,
        longitude
    )


    # ----------------------------------------------
    # MERGE AIR + WEATHER
    # ----------------------------------------------

    df = pd.merge(
        air_df,
        weather_df,
        on="time",
        how="inner"
    )

    df = (
        df
        .sort_values("time")
        .drop_duplicates(
            subset="time"
        )
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


    # ----------------------------------------------
    # AQI + TIME FEATURES
    # ----------------------------------------------

    df = add_aqi(
        df
    )

    df = add_time_features(
        df
    )


    # ----------------------------------------------
    # USE ONLY DATA AVAILABLE NOW
    # ----------------------------------------------

    now = (
        pd.Timestamp.now(
            tz="Asia/Kolkata"
        )
        .tz_localize(None)
        .floor("h")
    )


    history = df[
        df["time"] <= now
    ].copy()


    history = history.dropna(
        subset=SEQUENCE_FEATURES
    )


    if len(history) < 72:

        raise ValueError(
            "Not enough recent data "
            "for 72-hour LSTM input."
        )


    # Last 72 valid hours
    history = (
        history
        .tail(72)
        .copy()
    )


    # Ensure hourly continuity
    gaps = (
        history["time"]
        .diff()
        .dropna()
    )


    if not (
        gaps
        == pd.Timedelta(hours=1)
    ).all():

        raise ValueError(
            "Recent hourly data contains gaps."
        )


    current_time = (
        history["time"]
        .iloc[-1]
    )


    # ----------------------------------------------
    # CURRENT + OLD AQI VALUES
    # ----------------------------------------------

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


    # ----------------------------------------------
    # FUTURE WEATHER
    # ----------------------------------------------

    target_24_time = (
        current_time
        + pd.Timedelta(
            hours=24
        )
    )

    target_48_time = (
        current_time
        + pd.Timedelta(
            hours=48
        )
    )


    weather24 = get_future_weather(
        weather_df,
        target_24_time
    )

    weather48 = get_future_weather(
        weather_df,
        target_48_time
    )


    # ----------------------------------------------
    # SCALE 72-HOUR SEQUENCE
    # ----------------------------------------------

    sequence = (
        history[
            SEQUENCE_FEATURES
        ]
        .to_numpy(
            dtype=float
        )
    )


    scaled_sequence = (
        feature_scaler.transform(
            sequence
        )
    )


    X = np.expand_dims(
        scaled_sequence,
        axis=0
    )


    # ----------------------------------------------
    # AUXILIARY FEATURES
    # EXACT TRAINING ORDER
    # ----------------------------------------------

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


    auxiliary = (
        auxiliary
        .reshape(
            1,
            -1
        )
    )


    scaled_aux = (
        aux_scaler.transform(
            auxiliary
        )
    )


    # ----------------------------------------------
    # MODEL PREDICTION
    # ----------------------------------------------

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


    # Keep AQI inside valid prototype range
    prediction_24 = float(
        np.clip(
            prediction_24,
            0,
            500
        )
    )

    prediction_48 = float(
        np.clip(
            prediction_48,
            0,
            500
        )
    )


    prediction_24 = int(
        round(
            prediction_24
        )
    )

    prediction_48 = int(
        round(
            prediction_48
        )
    )


    return {

        "Current_AQI":
            int(
                round(
                    current_aqi
                )
            ),

        "Prediction_24h":
            prediction_24,

        "Category_24h":
            get_category(
                prediction_24
            ),

        "Prediction_48h":
            prediction_48,

        "Category_48h":
            get_category(
                prediction_48
            ),

        "Reference_Time":
            current_time,

        "Target_24h_Time":
            target_24_time,

        "Target_48h_Time":
            target_48_time,

        "Model":
            "V3 LSTM"
    }


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    # Belagavi coordinates
    result = predict_aqi(
        15.8521,
        74.5045
    )

    print(
        "\nLIVE V3 LSTM FORECAST\n"
    )

    print(
        "Current AQI:",
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