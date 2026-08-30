import requests
import pandas as pd
import os


locations = {
    "Belagavi": (15.8521, 74.5045),
    "Sambra": (15.8666, 74.6189),
    "Khanapur": (15.6394, 74.5083),
    "Bailhongal": (15.8130, 74.8580),
    "Gokak": (16.1691, 74.8338)
}


results = []


for name, (latitude, longitude) in locations.items():

    print(f"Fetching data for {name}...")

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
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
        "past_hours": 24,
        "forecast_hours": 1,
        "timezone": "Asia/Kolkata"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data["hourly"])


    # 24-hour averages
    pm10 = df["pm10"].tail(24).mean()
    pm25 = df["pm2_5"].tail(24).mean()

    no2 = df["nitrogen_dioxide"].tail(24).mean()
    so2 = df["sulphur_dioxide"].tail(24).mean()

    # 8-hour averages
    ozone = df["ozone"].tail(8).mean()

    co = (
        df["carbon_monoxide"]
        .tail(8)
        .mean()
        / 1000
    )


    results.append({
        "location": name,
        "latitude": latitude,
        "longitude": longitude,
        "pm2_5": round(pm25, 2),
        "pm10": round(pm10, 2),
        "no2": round(no2, 2),
        "so2": round(so2, 2),
        "ozone": round(ozone, 2),
        "co": round(co, 3)
    })


result_df = pd.DataFrame(results)


print("\nMultiple-location pollution data:\n")
print(result_df)


os.makedirs(
    "data/processed",
    exist_ok=True
)


result_df.to_csv(
    "data/processed/hotspot_locations.csv",
    index=False
)


print(
    "\nData saved to:"
    " data/processed/hotspot_locations.csv"
)