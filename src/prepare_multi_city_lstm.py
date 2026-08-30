import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

DATA_FILE = (
    "data/processed/"
    "multi_city_final_aqi_dataset.csv"
)

df = pd.read_csv(DATA_FILE)

df["time"] = pd.to_datetime(
    df["time"]
)


# --------------------------------------------------
# TIME FEATURES
# --------------------------------------------------

df["hour"] = (
    df["time"].dt.hour
)

df["day_of_year"] = (
    df["time"].dt.dayofyear
)


df["hour_sin"] = np.sin(
    2 * np.pi
    * df["hour"]
    / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi
    * df["hour"]
    / 24
)


df["day_sin"] = np.sin(
    2 * np.pi
    * df["day_of_year"]
    / 365
)

df["day_cos"] = np.cos(
    2 * np.pi
    * df["day_of_year"]
    / 365
)


# --------------------------------------------------
# SEQUENCE FEATURES
# Exact past-hour information
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


# --------------------------------------------------
# WEATHER FEATURES
# --------------------------------------------------

WEATHER_FEATURES = [

    "temperature_2m",

    "relative_humidity_2m",

    "precipitation",

    "wind_speed_10m"
]


LOOKBACK = 72

FORECAST_24 = 24

FORECAST_48 = 48


# --------------------------------------------------
# CREATE CHRONOLOGICAL SPLIT FOR EACH CITY
# --------------------------------------------------

city_split = {}


for city in df["city"].unique():

    city_rows = (
        df[df["city"] == city]
        .sort_values("time")
        .reset_index(drop=True)
    )

    city_split[city] = int(
        len(city_rows) * 0.80
    )


# --------------------------------------------------
# FIT SEQUENCE SCALER
# TRAINING DATA ONLY
# --------------------------------------------------

training_feature_frames = []


for city in df["city"].unique():

    city_df = (
        df[df["city"] == city]
        .sort_values("time")
        .reset_index(drop=True)
    )

    split_index = (
        city_split[city]
    )

    training_feature_frames.append(

        city_df[
            SEQUENCE_FEATURES
        ].iloc[:split_index]
    )


training_features = pd.concat(
    training_feature_frames,
    ignore_index=True
)


feature_scaler = MinMaxScaler()


feature_scaler.fit(
    training_features
)


# --------------------------------------------------
# SAMPLE STORAGE
# --------------------------------------------------

X_train = []
aux_train_raw = []
y_train_raw = []


X_test = []
aux_test_raw = []
y_test_raw = []


current_test = []

city_test = []

latitude_test = []

longitude_test = []


# --------------------------------------------------
# BUILD SAMPLES CITY BY CITY
# --------------------------------------------------

for city in df["city"].unique():

    print(
        f"Preparing {city}..."
    )


    city_df = (
        df[
            df["city"] == city
        ]
        .copy()
        .sort_values("time")
        .reset_index(drop=True)
    )


    split_index = (
        city_split[city]
    )


    # ----------------------------------------------
    # SCALE SEQUENCE
    # ----------------------------------------------

    scaled_sequence = (
        feature_scaler.transform(

            city_df[
                SEQUENCE_FEATURES
            ]
        )
    )


    train_count = 0
    test_count = 0


    for i in range(
        LOOKBACK,
        len(city_df)
        - FORECAST_48
        + 1
    ):


        future24_index = (
            i
            + FORECAST_24
            - 1
        )


        future48_index = (
            i
            + FORECAST_48
            - 1
        )


        # ------------------------------------------
        # LAST 72 HOURS
        # ------------------------------------------

        sequence = (
            scaled_sequence[
                i - LOOKBACK:i
            ]
        )


        # ------------------------------------------
        # AQI HISTORY
        # ------------------------------------------

        current_aqi = float(
            city_df["AQI"]
            .iloc[i - 1]
        )


        aqi_24_ago = float(
            city_df["AQI"]
            .iloc[i - 25]
        )


        aqi_48_ago = float(
            city_df["AQI"]
            .iloc[i - 49]
        )


        # ------------------------------------------
        # FUTURE AQI
        # ------------------------------------------

        future24_aqi = float(
            city_df["AQI"]
            .iloc[future24_index]
        )


        future48_aqi = float(
            city_df["AQI"]
            .iloc[future48_index]
        )


        # ------------------------------------------
        # FUTURE WEATHER
        # ------------------------------------------

        weather24 = (

            city_df[
                WEATHER_FEATURES
            ]
            .iloc[
                future24_index
            ]
            .to_numpy(
                dtype=float
            )
        )


        weather48 = (

            city_df[
                WEATHER_FEATURES
            ]
            .iloc[
                future48_index
            ]
            .to_numpy(
                dtype=float
            )
        )


        # ------------------------------------------
        # AQI TREND
        # ------------------------------------------

        trend24 = (
            current_aqi
            - aqi_24_ago
        )


        previous_trend24 = (
            aqi_24_ago
            - aqi_48_ago
        )


        # ------------------------------------------
        # LOCATION
        # ------------------------------------------

        latitude = float(
            city_df["latitude"]
            .iloc[i - 1]
        )


        longitude = float(
            city_df["longitude"]
            .iloc[i - 1]
        )


        # ------------------------------------------
        # AUXILIARY INPUT
        #
        # 4 weather24
        # 4 weather48
        # 5 AQI/trend
        # 2 coordinates
        #
        # Total = 15
        # ------------------------------------------

        auxiliary = np.concatenate([

            weather24,

            weather48,

            [
                current_aqi,

                aqi_24_ago,

                aqi_48_ago,

                trend24,

                previous_trend24,

                latitude,

                longitude
            ]
        ])


        # ------------------------------------------
        # RESIDUAL TARGET
        # ------------------------------------------

        target = [

            future24_aqi
            - current_aqi,

            future48_aqi
            - current_aqi
        ]


        # ------------------------------------------
        # TRAIN
        # Future target must remain inside train
        # ------------------------------------------

        if future48_index < split_index:

            X_train.append(
                sequence
            )

            aux_train_raw.append(
                auxiliary
            )

            y_train_raw.append(
                target
            )

            train_count += 1


        # ------------------------------------------
        # TEST
        # ------------------------------------------

        elif i >= split_index:

            X_test.append(
                sequence
            )

            aux_test_raw.append(
                auxiliary
            )

            y_test_raw.append(
                target
            )

            current_test.append(
                current_aqi
            )

            city_test.append(
                city
            )

            latitude_test.append(
                latitude
            )

            longitude_test.append(
                longitude
            )

            test_count += 1


    print(
        f"   Train samples: {train_count}"
    )

    print(
        f"   Test samples: {test_count}"
    )


# --------------------------------------------------
# CONVERT TO NUMPY
# --------------------------------------------------

X_train = np.array(
    X_train,
    dtype=np.float32
)

X_test = np.array(
    X_test,
    dtype=np.float32
)


aux_train_raw = np.array(
    aux_train_raw,
    dtype=np.float32
)

aux_test_raw = np.array(
    aux_test_raw,
    dtype=np.float32
)


y_train_raw = np.array(
    y_train_raw,
    dtype=np.float32
)

y_test_raw = np.array(
    y_test_raw,
    dtype=np.float32
)


current_test = np.array(
    current_test,
    dtype=np.float32
)


city_test = np.array(
    city_test
)


latitude_test = np.array(
    latitude_test,
    dtype=np.float32
)


longitude_test = np.array(
    longitude_test,
    dtype=np.float32
)


# --------------------------------------------------
# SCALE AUXILIARY INPUT
# --------------------------------------------------

aux_scaler = MinMaxScaler()


aux_train = (
    aux_scaler.fit_transform(
        aux_train_raw
    )
)


aux_test = (
    aux_scaler.transform(
        aux_test_raw
    )
)


# --------------------------------------------------
# SCALE TARGET DELTAS
# --------------------------------------------------

target_scaler = StandardScaler()


y_train = (
    target_scaler.fit_transform(
        y_train_raw
    )
)


y_test = (
    target_scaler.transform(
        y_test_raw
    )
)


# --------------------------------------------------
# CREATE DIRECTORIES
# --------------------------------------------------

os.makedirs(
    "data/model_ready_multi_city",
    exist_ok=True
)


os.makedirs(
    "models",
    exist_ok=True
)


# --------------------------------------------------
# SAVE DATA
# --------------------------------------------------

np.save(
    "data/model_ready_multi_city/X_train.npy",
    X_train
)

np.save(
    "data/model_ready_multi_city/X_test.npy",
    X_test
)


np.save(
    "data/model_ready_multi_city/aux_train.npy",
    aux_train
)

np.save(
    "data/model_ready_multi_city/aux_test.npy",
    aux_test
)


np.save(
    "data/model_ready_multi_city/y_train.npy",
    y_train
)

np.save(
    "data/model_ready_multi_city/y_test.npy",
    y_test
)


np.save(
    "data/model_ready_multi_city/current_test.npy",
    current_test
)


np.save(
    "data/model_ready_multi_city/city_test.npy",
    city_test
)


np.save(
    "data/model_ready_multi_city/latitude_test.npy",
    latitude_test
)


np.save(
    "data/model_ready_multi_city/longitude_test.npy",
    longitude_test
)


# --------------------------------------------------
# SAVE SCALERS
# --------------------------------------------------

joblib.dump(
    feature_scaler,
    "models/multi_feature_scaler.pkl"
)


joblib.dump(
    aux_scaler,
    "models/multi_aux_scaler.pkl"
)


joblib.dump(
    target_scaler,
    "models/multi_target_scaler.pkl"
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print(
    "\n================================="
)

print(
    "MULTI-CITY LSTM DATA READY"
)

print(
    "================================="
)


print(
    "X_train:",
    X_train.shape
)


print(
    "Aux train:",
    aux_train.shape
)


print(
    "y_train:",
    y_train.shape
)


print(
    "\nX_test:",
    X_test.shape
)


print(
    "Aux test:",
    aux_test.shape
)


print(
    "y_test:",
    y_test.shape
)


print(
    "\nTraining cities:",
    df["city"].nunique()
)


print(
    "Sequence features:",
    len(SEQUENCE_FEATURES)
)


print(
    "Auxiliary features:",
    aux_train.shape[1]
)