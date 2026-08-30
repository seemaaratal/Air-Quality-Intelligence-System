import pandas as pd

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/processed/air_quality_weather.csv")
df["time"] = pd.to_datetime(df["time"])


# --------------------------------------------------
# AQI ROLLING AVERAGES
# --------------------------------------------------

# CPCB uses 24-hour average for these pollutants
df["pm10_avg"] = df["pm10"].rolling(
    window=24,
    min_periods=24
).mean()

df["pm2_5_avg"] = df["pm2_5"].rolling(
    window=24,
    min_periods=24
).mean()

df["no2_avg"] = df["nitrogen_dioxide"].rolling(
    window=24,
    min_periods=24
).mean()

df["so2_avg"] = df["sulphur_dioxide"].rolling(
    window=24,
    min_periods=24
).mean()


# CPCB uses 8-hour average for O3 and CO
df["o3_avg"] = df["ozone"].rolling(
    window=8,
    min_periods=8
).mean()

# Open-Meteo gives CO in µg/m³.
# Convert to mg/m³ for CPCB CO breakpoints.
df["co_mg"] = df["carbon_monoxide"] / 1000

df["co_avg"] = df["co_mg"].rolling(
    window=8,
    min_periods=8
).mean()


# --------------------------------------------------
# CPCB SUB-INDEX FUNCTION
# --------------------------------------------------

def calculate_sub_index(value, breakpoints):

    if pd.isna(value):
        return None

    # Find the breakpoint interval
    for low, high, index_low, index_high in breakpoints:

        if low <= value <= high:

            sub_index = (
                ((index_high - index_low) / (high - low))
                * (value - low)
                + index_low
            )

            return round(sub_index)

    # Above maximum CPCB range
    if value > breakpoints[-1][1]:
        return 500

    return None


# --------------------------------------------------
# CPCB BREAKPOINTS
# Continuous boundaries are used here so decimal
# rolling averages do not fall into artificial gaps.
# --------------------------------------------------

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
    (0, 1.0, 0, 50),
    (1.0, 2.0, 51, 100),
    (2.0, 10.0, 101, 200),
    (10.0, 17.0, 201, 300),
    (17.0, 34.0, 301, 400),
    (34.0, 100.0, 401, 500)
]


# --------------------------------------------------
# CALCULATE SUB-INDICES
# --------------------------------------------------

df["aqi_pm10"] = df["pm10_avg"].apply(
    lambda x: calculate_sub_index(x, pm10_bp)
)

df["aqi_pm25"] = df["pm2_5_avg"].apply(
    lambda x: calculate_sub_index(x, pm25_bp)
)

df["aqi_no2"] = df["no2_avg"].apply(
    lambda x: calculate_sub_index(x, no2_bp)
)

df["aqi_so2"] = df["so2_avg"].apply(
    lambda x: calculate_sub_index(x, so2_bp)
)

df["aqi_o3"] = df["o3_avg"].apply(
    lambda x: calculate_sub_index(x, o3_bp)
)

df["aqi_co"] = df["co_avg"].apply(
    lambda x: calculate_sub_index(x, co_bp)
)


# --------------------------------------------------
# REMOVE ROWS WITHOUT COMPLETE 24-HOUR HISTORY
# --------------------------------------------------

required_columns = [
    "aqi_pm10",
    "aqi_pm25",
    "aqi_no2",
    "aqi_so2"
]

df = df.dropna(
    subset=required_columns
).reset_index(drop=True)


# --------------------------------------------------
# OVERALL AQI
# --------------------------------------------------

aqi_columns = [
    "aqi_pm10",
    "aqi_pm25",
    "aqi_no2",
    "aqi_so2",
    "aqi_o3",
    "aqi_co"
]

df["AQI"] = df[aqi_columns].max(axis=1)


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


df["AQI_Category"] = df["AQI"].apply(get_category)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

output_file = "data/processed/final_aqi_dataset.csv"

df.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

print("\nPreprocessing completed!")

print("\nTotal rows:", len(df))

print("\nFirst valid timestamp:")
print(df["time"].iloc[0])

print("\nAQI sample:")
print(
    df[
        [
            "time",
            "pm2_5_avg",
            "pm10_avg",
            "AQI",
            "AQI_Category"
        ]
    ].head(10)
)

print("\nAQI statistics:")
print(df["AQI"].describe())

print("\nAQI categories:")
print(df["AQI_Category"].value_counts())

print("\nDataset saved:")
print(output_file)