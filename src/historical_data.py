import requests
import pandas as pd
import os


# --------------------------------------------------
# LOCATION
# --------------------------------------------------

latitude = 15.85212
longitude = 74.50447

# One year of historical data
start_date = "2025-08-28"
end_date = "2026-08-27"


print("Downloading historical air-quality data...")


# --------------------------------------------------
# AIR QUALITY DATA
# --------------------------------------------------

air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

air_params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": [
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone"
    ],
    "timezone": "Asia/Kolkata"
}


air_response = requests.get(
    air_url,
    params=air_params,
    timeout=60
)

air_response.raise_for_status()

air_json = air_response.json()

air_df = pd.DataFrame(air_json["hourly"])


print("Air-quality rows:", len(air_df))


# --------------------------------------------------
# HISTORICAL WEATHER DATA
# --------------------------------------------------

print("Downloading historical weather data...")


weather_url = "https://archive-api.open-meteo.com/v1/archive"

weather_params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m"
    ],
    "timezone": "Asia/Kolkata"
}


weather_response = requests.get(
    weather_url,
    params=weather_params,
    timeout=60
)

weather_response.raise_for_status()

weather_json = weather_response.json()

weather_df = pd.DataFrame(weather_json["hourly"])


print("Weather rows:", len(weather_df))


# --------------------------------------------------
# CONVERT TIME
# --------------------------------------------------

air_df["time"] = pd.to_datetime(air_df["time"])

weather_df["time"] = pd.to_datetime(weather_df["time"])


# --------------------------------------------------
# MERGE BOTH DATASETS
# --------------------------------------------------

combined_df = pd.merge(
    air_df,
    weather_df,
    on="time",
    how="inner"
)


# --------------------------------------------------
# SORT DATA
# --------------------------------------------------

combined_df = combined_df.sort_values("time")

combined_df.reset_index(drop=True, inplace=True)


# --------------------------------------------------
# DISPLAY INFORMATION
# --------------------------------------------------

print("\nCombined dataset:")
print(combined_df.head())

print("\nTotal rows:", len(combined_df))

print("\nColumns:")
print(combined_df.columns.tolist())

print("\nMissing values:")
print(combined_df.isnull().sum())


# --------------------------------------------------
# SAVE DATA
# --------------------------------------------------

os.makedirs("data/processed", exist_ok=True)

output_file = "data/processed/air_quality_weather.csv"

combined_df.to_csv(
    output_file,
    index=False
)


print("\nDataset successfully saved!")
print("File:", output_file)