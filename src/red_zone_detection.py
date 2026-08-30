import pandas as pd


# Load AQI data
df = pd.read_csv(
    "data/processed/hotspot_aqi.csv"
)


# ------------------------------------------
# HOTSPOT / RED ZONE LOGIC
# ------------------------------------------

def detect_zone(aqi):

    if aqi <= 100:
        return "Normal"

    elif aqi <= 200:
        return "Pollution Hotspot"

    elif aqi <= 300:
        return "Red Zone"

    else:
        return "Critical Red Zone"


df["Zone_Status"] = df["AQI"].apply(
    detect_zone
)


# ------------------------------------------
# HOTSPOT FLAG
# ------------------------------------------

df["Is_Hotspot"] = df["AQI"] > 100

df["Is_Red_Zone"] = df["AQI"] > 200


# Save result
df.to_csv(
    "data/processed/red_zone_results.csv",
    index=False
)


print("\nHOTSPOT / RED-ZONE RESULTS:\n")

print(
    df[
        [
            "location",
            "AQI",
            "AQI_Category",
            "Zone_Status",
            "Is_Hotspot",
            "Is_Red_Zone"
        ]
    ]
)


print("\nTotal hotspots:", df["Is_Hotspot"].sum())

print(
    "Total red zones:",
    df["Is_Red_Zone"].sum()
)

print(
    "\nSaved to:"
    " data/processed/red_zone_results.csv"
)