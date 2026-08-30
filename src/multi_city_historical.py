import os
import time
import requests
import pandas as pd


# --------------------------------------------------
# DATE RANGE
# Same historical period used earlier
# --------------------------------------------------

START_DATE = "2025-08-28"
END_DATE = "2026-08-27"


# --------------------------------------------------
# TRAINING CITIES
# --------------------------------------------------

CITIES = {

    "Belagavi": {
        "latitude": 15.8521,
        "longitude": 74.5045
    },

    "Hubballi": {
        "latitude": 15.3478,
        "longitude": 75.1338
    },

    "Bengaluru": {
        "latitude": 12.9716,
        "longitude": 77.5946
    },

    "Delhi": {
        "latitude": 28.6139,
        "longitude": 77.2090
    },

    "Mumbai": {
        "latitude": 19.0760,
        "longitude": 72.8777
    },

    "Pune": {
        "latitude": 18.5204,
        "longitude": 73.8567
    },

    "Hyderabad": {
        "latitude": 17.3850,
        "longitude": 78.4867
    },

    "Chennai": {
        "latitude": 13.0827,
        "longitude": 80.2707
    },

    "Kolkata": {
        "latitude": 22.5726,
        "longitude": 88.3639
    },

    "Ahmedabad": {
        "latitude": 23.0225,
        "longitude": 72.5714
    }
}


# --------------------------------------------------
# API URLS
# --------------------------------------------------

AIR_URL = (
    "https://air-quality-api.open-meteo.com/"
    "v1/air-quality"
)

WEATHER_URL = (
    "https://archive-api.open-meteo.com/"
    "v1/archive"
)


# --------------------------------------------------
# OUTPUT DIRECTORY
# --------------------------------------------------

OUTPUT_DIR = (
    "data/processed/multi_city"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# --------------------------------------------------
# REQUEST WITH RETRIES
# --------------------------------------------------

def get_with_retry(
    url,
    params,
    attempts=3
):

    last_error = None

    for attempt in range(
        1,
        attempts + 1
    ):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=(20, 90)
            )

            response.raise_for_status()

            return response.json()

        except Exception as error:

            last_error = error

            print(
                f"   Attempt {attempt} failed:"
                f" {type(error).__name__}"
            )

            if attempt < attempts:

                print(
                    "   Retrying..."
                )

                time.sleep(5)

    raise last_error


# --------------------------------------------------
# FETCH ONE CITY
# --------------------------------------------------

def fetch_city_data(
    city,
    latitude,
    longitude
):

    print(
        f"\nFetching data for {city}..."
    )


    # ----------------------------------------------
    # AIR QUALITY
    # ----------------------------------------------

    air_params = {

        "latitude":
            latitude,

        "longitude":
            longitude,

        "start_date":
            START_DATE,

        "end_date":
            END_DATE,

        "hourly": (
            "pm10,"
            "pm2_5,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone"
        ),

        "timezone":
            "Asia/Kolkata"
    }


    air_json = get_with_retry(
        AIR_URL,
        air_params
    )


    air_df = pd.DataFrame(
        air_json["hourly"]
    )


    air_df["time"] = pd.to_datetime(
        air_df["time"]
    )


    print(
        "   Air-quality rows:",
        len(air_df)
    )


    # ----------------------------------------------
    # WEATHER
    # ----------------------------------------------

    weather_params = {

        "latitude":
            latitude,

        "longitude":
            longitude,

        "start_date":
            START_DATE,

        "end_date":
            END_DATE,

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m"
        ),

        "timezone":
            "Asia/Kolkata"
    }


    weather_json = get_with_retry(
        WEATHER_URL,
        weather_params
    )


    weather_df = pd.DataFrame(
        weather_json["hourly"]
    )


    weather_df["time"] = pd.to_datetime(
        weather_df["time"]
    )


    print(
        "   Weather rows:",
        len(weather_df)
    )


    # ----------------------------------------------
    # MERGE
    # ----------------------------------------------

    merged = pd.merge(
        air_df,
        weather_df,
        on="time",
        how="inner"
    )


    merged = (
        merged
        .sort_values("time")
        .drop_duplicates(
            subset="time"
        )
        .reset_index(drop=True)
    )


    # ----------------------------------------------
    # CITY INFORMATION
    # ----------------------------------------------

    merged.insert(
        0,
        "city",
        city
    )


    merged.insert(
        1,
        "latitude",
        latitude
    )


    merged.insert(
        2,
        "longitude",
        longitude
    )


    print(
        "   Final merged rows:",
        len(merged)
    )


    # ----------------------------------------------
    # SAVE INDIVIDUAL CITY
    # ----------------------------------------------

    filename = (
        city
        .lower()
        .replace(" ", "_")
        + ".csv"
    )


    city_path = os.path.join(
        OUTPUT_DIR,
        filename
    )


    merged.to_csv(
        city_path,
        index=False
    )


    print(
        "   Saved:",
        city_path
    )


    return merged


# --------------------------------------------------
# MAIN
# --------------------------------------------------

all_city_data = []


for city, coordinates in CITIES.items():

    try:

        city_df = fetch_city_data(

            city,

            coordinates[
                "latitude"
            ],

            coordinates[
                "longitude"
            ]
        )


        all_city_data.append(
            city_df
        )


    except Exception as error:

        print(
            f"\n❌ Failed for {city}: "
            f"{error}"
        )


    # Small gap between cities
    time.sleep(2)


# --------------------------------------------------
# COMBINE EVERYTHING
# --------------------------------------------------

if len(all_city_data) == 0:

    print(
        "\nNo city data downloaded."
    )

else:

    final_df = pd.concat(
        all_city_data,
        ignore_index=True
    )


    final_path = (
        "data/processed/"
        "multi_city_air_quality_weather.csv"
    )


    final_df.to_csv(
        final_path,
        index=False
    )


    print(
        "\n================================="
    )

    print(
        "MULTI-CITY DATASET COMPLETE"
    )

    print(
        "================================="
    )


    print(
        "Cities downloaded:",
        final_df["city"].nunique()
    )


    print(
        "City names:",
        final_df[
            "city"
        ].unique().tolist()
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
        "Saved:",
        final_path
    )