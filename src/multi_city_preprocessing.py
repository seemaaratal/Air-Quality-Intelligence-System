import pandas as pd
import numpy as np
import os

from src.live_air_quality import calculate_aqi


# --------------------------------------------------
# LOAD MULTI-CITY DATA
# --------------------------------------------------

INPUT_FILE = (
    "data/processed/"
    "multi_city_air_quality_weather.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "multi_city_final_aqi_dataset.csv"
)


df = pd.read_csv(
    INPUT_FILE
)

df["time"] = pd.to_datetime(
    df["time"]
)


print("\nMULTI-CITY AQI PREPROCESSING\n")

print(
    "Original rows:",
    len(df)
)

print(
    "Cities:",
    df["city"].nunique()
)


# --------------------------------------------------
# PROCESS ONE CITY AT A TIME
# --------------------------------------------------

processed_cities = []


for city in df["city"].unique():

    print(
        f"\nProcessing {city}..."
    )


    city_df = (
        df[
            df["city"] == city
        ]
        .copy()
        .sort_values("time")
        .reset_index(drop=True)
    )


    # --------------------------------------------------
    # 24-HOUR AVERAGES
    # --------------------------------------------------

    city_df["pm25_24h"] = (
        city_df["pm2_5"]
        .rolling(
            window=24,
            min_periods=24
        )
        .mean()
    )


    city_df["pm10_24h"] = (
        city_df["pm10"]
        .rolling(
            window=24,
            min_periods=24
        )
        .mean()
    )


    city_df["no2_24h"] = (
        city_df["nitrogen_dioxide"]
        .rolling(
            window=24,
            min_periods=24
        )
        .mean()
    )


    city_df["so2_24h"] = (
        city_df["sulphur_dioxide"]
        .rolling(
            window=24,
            min_periods=24
        )
        .mean()
    )


    # --------------------------------------------------
    # 8-HOUR AVERAGES
    # --------------------------------------------------

    city_df["ozone_8h"] = (
        city_df["ozone"]
        .rolling(
            window=8,
            min_periods=8
        )
        .mean()
    )


    city_df["co_8h"] = (
        city_df["carbon_monoxide"]
        .rolling(
            window=8,
            min_periods=8
        )
        .mean()
        / 1000
    )


    # --------------------------------------------------
    # REMOVE ROWS WITHOUT FULL AQI WINDOW
    # --------------------------------------------------

    required_columns = [
        "pm25_24h",
        "pm10_24h",
        "no2_24h",
        "so2_24h",
        "ozone_8h",
        "co_8h"
    ]


    city_df = (
        city_df
        .dropna(
            subset=required_columns
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------
    # CALCULATE AQI
    # --------------------------------------------------

    aqi_values = []
    categories = []
    dominant_pollutants = []


    for _, row in city_df.iterrows():

        (
            aqi,
            category,
            dominant

        ) = calculate_aqi(

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

        categories.append(
            category
        )

        dominant_pollutants.append(
            dominant
        )


    city_df["AQI"] = (
        aqi_values
    )

    city_df["AQI_Category"] = (
        categories
    )

    city_df["Dominant_Pollutant"] = (
        dominant_pollutants
    )


    processed_cities.append(
        city_df
    )


    print(
        "   Final rows:",
        len(city_df)
    )

    print(
        "   AQI min:",
        city_df["AQI"].min()
    )

    print(
        "   AQI mean:",
        round(
            city_df["AQI"].mean(),
            2
        )
    )

    print(
        "   AQI max:",
        city_df["AQI"].max()
    )


# --------------------------------------------------
# COMBINE ALL CITIES
# --------------------------------------------------

final_df = pd.concat(
    processed_cities,
    ignore_index=True
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print(
    "\n================================="
)

print(
    "MULTI-CITY AQI DATASET COMPLETE"
)

print(
    "================================="
)


print(
    "Cities:",
    final_df["city"].nunique()
)


print(
    "Total rows:",
    len(final_df)
)


print(
    "Missing values:",
    int(
        final_df
        .isna()
        .sum()
        .sum()
    )
)


print(
    "\nAQI category counts:"
)

print(
    final_df[
        "AQI_Category"
    ].value_counts()
)


print(
    "\nAverage AQI by city:"
)

print(
    final_df
    .groupby("city")["AQI"]
    .mean()
    .round(2)
    .sort_values(
        ascending=False
    )
)


print(
    "\nSaved:",
    OUTPUT_FILE
)