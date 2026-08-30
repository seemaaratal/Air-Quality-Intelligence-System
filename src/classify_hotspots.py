import pandas as pd


# Load Step 1 data
df = pd.read_csv(
    "data/processed/hotspot_locations.csv"
)


# ------------------------------------------
# AQI SUB-INDEX FUNCTION
# ------------------------------------------

def calculate_sub_index(value, breakpoints):

    for low, high, index_low, index_high in breakpoints:

        if low <= value <= high:

            aqi = (
                ((index_high - index_low) / (high - low))
                * (value - low)
                + index_low
            )

            return round(aqi)

    return 500


# CPCB-style breakpoints

pm10_bp = [
    (0, 50, 0, 50),
    (50, 100, 51, 100),
    (100, 250, 101, 200),
    (250, 350, 201, 300),
    (350, 430, 301, 400),
    (430, 1000, 401, 500)
]

pm25_bp = [
    (0, 30, 0, 50),
    (30, 60, 51, 100),
    (60, 90, 101, 200),
    (90, 120, 201, 300),
    (120, 250, 301, 400),
    (250, 500, 401, 500)
]

no2_bp = [
    (0, 40, 0, 50),
    (40, 80, 51, 100),
    (80, 180, 101, 200),
    (180, 280, 201, 300),
    (280, 400, 301, 400),
    (400, 1000, 401, 500)
]

so2_bp = [
    (0, 40, 0, 50),
    (40, 80, 51, 100),
    (80, 380, 101, 200),
    (380, 800, 201, 300),
    (800, 1600, 301, 400),
    (1600, 3000, 401, 500)
]

o3_bp = [
    (0, 50, 0, 50),
    (50, 100, 51, 100),
    (100, 168, 101, 200),
    (168, 208, 201, 300),
    (208, 748, 301, 400),
    (748, 1500, 401, 500)
]

co_bp = [
    (0, 1, 0, 50),
    (1, 2, 51, 100),
    (2, 10, 101, 200),
    (10, 17, 201, 300),
    (17, 34, 301, 400),
    (34, 100, 401, 500)
]


# ------------------------------------------
# CALCULATE SUB-INDICES
# ------------------------------------------

df["aqi_pm25"] = df["pm2_5"].apply(
    lambda x: calculate_sub_index(x, pm25_bp)
)

df["aqi_pm10"] = df["pm10"].apply(
    lambda x: calculate_sub_index(x, pm10_bp)
)

df["aqi_no2"] = df["no2"].apply(
    lambda x: calculate_sub_index(x, no2_bp)
)

df["aqi_so2"] = df["so2"].apply(
    lambda x: calculate_sub_index(x, so2_bp)
)

df["aqi_o3"] = df["ozone"].apply(
    lambda x: calculate_sub_index(x, o3_bp)
)

df["aqi_co"] = df["co"].apply(
    lambda x: calculate_sub_index(x, co_bp)
)


# Overall AQI = highest pollutant sub-index

aqi_columns = [
    "aqi_pm25",
    "aqi_pm10",
    "aqi_no2",
    "aqi_so2",
    "aqi_o3",
    "aqi_co"
]

df["AQI"] = df[aqi_columns].max(axis=1)


# ------------------------------------------
# AQI CATEGORY
# ------------------------------------------

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


df["AQI_Category"] = df["AQI"].apply(
    get_category
)


# Save
df.to_csv(
    "data/processed/hotspot_aqi.csv",
    index=False
)


print("\nAQI Classification:\n")

print(
    df[
        [
            "location",
            "AQI",
            "AQI_Category"
        ]
    ]
)

print(
    "\nSaved to:"
    " data/processed/hotspot_aqi.csv"
)