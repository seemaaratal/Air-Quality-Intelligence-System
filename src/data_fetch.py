import requests
import pandas as pd
import os


# Step 1: Enter city name
city = input("Enter city name: ")


# Step 2: Convert city name into latitude and longitude
geo_url = "https://geocoding-api.open-meteo.com/v1/search"

geo_params = {
    "name": city,
    "count": 1,
    "language": "en",
    "format": "json"
}

geo_response = requests.get(geo_url, params=geo_params, timeout=10)
geo_response.raise_for_status()

geo_data = geo_response.json()

if "results" not in geo_data:
    print("City not found!")
    exit()

latitude = geo_data["results"][0]["latitude"]
longitude = geo_data["results"][0]["longitude"]
location_name = geo_data["results"][0]["name"]

print("\nLocation:", location_name)
print("Latitude:", latitude)
print("Longitude:", longitude)


# Step 3: Fetch air-quality data
air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

air_params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly": [
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone"
    ],
    "timezone": "auto"
}

air_response = requests.get(air_url, params=air_params, timeout=10)
air_response.raise_for_status()

air_data = air_response.json()


# Step 4: Convert JSON data into DataFrame
df = pd.DataFrame(air_data["hourly"])

print("\nAir Quality Data:")
print(df.head())


# Step 5: Save data into CSV
os.makedirs("data/raw", exist_ok=True)

file_path = "data/raw/air_quality_data.csv"

df.to_csv(file_path, index=False)

print("\nData successfully saved!")
print("File:", file_path)