import requests
import pandas as pd


def sub_index(value, breakpoints):
    if pd.isna(value):
        return None

    for low, high, i_low, i_high in breakpoints:
        if low <= value <= high:
            return round(
                ((i_high - i_low) / (high - low))
                * (value - low)
                + i_low
            )

    return 500


def calculate_aqi(pm25, pm10, no2, so2, ozone, co):

    pm25_bp = [
        (0, 30, 0, 50),
        (30, 60, 50, 100),
        (60, 90, 100, 200),
        (90, 120, 200, 300),
        (120, 250, 300, 400),
        (250, 500, 400, 500)
    ]

    pm10_bp = [
        (0, 50, 0, 50),
        (50, 100, 50, 100),
        (100, 250, 100, 200),
        (250, 350, 200, 300),
        (350, 430, 300, 400),
        (430, 600, 400, 500)
    ]

    no2_bp = [
        (0, 40, 0, 50),
        (40, 80, 50, 100),
        (80, 180, 100, 200),
        (180, 280, 200, 300),
        (280, 400, 300, 400),
        (400, 800, 400, 500)
    ]

    so2_bp = [
        (0, 40, 0, 50),
        (40, 80, 50, 100),
        (80, 380, 100, 200),
        (380, 800, 200, 300),
        (800, 1600, 300, 400),
        (1600, 2000, 400, 500)
    ]

    ozone_bp = [
        (0, 50, 0, 50),
        (50, 100, 50, 100),
        (100, 168, 100, 200),
        (168, 208, 200, 300),
        (208, 748, 300, 400),
        (748, 1000, 400, 500)
    ]

    co_bp = [
        (0, 1, 0, 50),
        (1, 2, 50, 100),
        (2, 10, 100, 200),
        (10, 17, 200, 300),
        (17, 34, 300, 400),
        (34, 50, 400, 500)
    ]

    indices = {
        "PM2.5": sub_index(pm25, pm25_bp),
        "PM10": sub_index(pm10, pm10_bp),
        "NO2": sub_index(no2, no2_bp),
        "SO2": sub_index(so2, so2_bp),
        "Ozone": sub_index(ozone, ozone_bp),
        "CO": sub_index(co, co_bp)
    }

    aqi = max(indices.values())
    dominant = max(indices, key=indices.get)

    if aqi <= 50:
        category = "Good"
    elif aqi <= 100:
        category = "Satisfactory"
    elif aqi <= 200:
        category = "Moderately Polluted"
    elif aqi <= 300:
        category = "Poor"
    elif aqi <= 400:
        category = "Very Poor"
    else:
        category = "Severe"

    return aqi, category, dominant


def get_live_air_quality(latitude, longitude):

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "pm10,pm2_5,carbon_monoxide,"
            "nitrogen_dioxide,sulphur_dioxide,ozone"
        ),
        "past_hours": 30,
        "forecast_hours": 1,
        "timezone": "Asia/Kolkata"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()["hourly"]

    df = pd.DataFrame(data)

    df["time"] = pd.to_datetime(df["time"])

    now = pd.Timestamp.now(
        tz="Asia/Kolkata"
    ).tz_localize(None).floor("h")

    df = df[df["time"] <= now]

    last24 = df.tail(24)
    last8 = df.tail(8)

    pm25 = last24["pm2_5"].mean()
    pm10 = last24["pm10"].mean()

    no2 = last24["nitrogen_dioxide"].mean()
    so2 = last24["sulphur_dioxide"].mean()

    ozone = last8["ozone"].mean()

    # µg/m³ → mg/m³
    co = last8["carbon_monoxide"].mean() / 1000

    aqi, category, dominant = calculate_aqi(
        pm25,
        pm10,
        no2,
        so2,
        ozone,
        co
    )

    return {
        "AQI": aqi,
        "Category": category,
        "Dominant_Pollutant": dominant,
        "PM2.5": round(pm25, 2),
        "PM10": round(pm10, 2),
        "NO2": round(no2, 2),
        "SO2": round(so2, 2),
        "Ozone": round(ozone, 2),
        "CO": round(co, 3),
        "Timestamp": df["time"].iloc[-1]
    }


if __name__ == "__main__":

    # Belagavi
    result = get_live_air_quality(
        15.8521,
        74.5045
    )

    print("\nCURRENT AIR QUALITY\n")

    for key, value in result.items():
        print(f"{key}: {value}")