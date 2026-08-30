import pandas as pd


# --------------------------------------------------
# LOAD AQI DATA
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/hotspot_aqi.csv"
)


# --------------------------------------------------
# DOMINANT POLLUTANT
# --------------------------------------------------

pollutant_columns = {
    "aqi_pm25": "PM2.5",
    "aqi_pm10": "PM10",
    "aqi_no2": "NO2",
    "aqi_so2": "SO2",
    "aqi_o3": "Ozone",
    "aqi_co": "CO"
}


def find_dominant_pollutant(row):

    highest_column = max(
        pollutant_columns.keys(),
        key=lambda col: row[col]
    )

    return pollutant_columns[highest_column]


df["Dominant_Pollutant"] = df.apply(
    find_dominant_pollutant,
    axis=1
)


# --------------------------------------------------
# AQI-BASED RECOMMENDATION
# --------------------------------------------------

def get_aqi_recommendation(aqi):

    if aqi <= 50:

        return (
            "Air quality is good. "
            "Continue regular monitoring and maintain green cover."
        )

    elif aqi <= 100:

        return (
            "Monitor pollution levels and traffic flow. "
            "Encourage public transport and maintain roadside greenery."
        )

    elif aqi <= 200:

        return (
            "Pollution hotspot detected. "
            "Increase traffic monitoring, control road dust "
            "and avoid open waste burning."
        )

    elif aqi <= 300:

        return (
            "Red zone detected. Issue pollution alert, "
            "strengthen traffic control and construction dust management."
        )

    elif aqi <= 400:

        return (
            "Very high pollution. Apply strict traffic and dust-control "
            "measures and increase environmental monitoring."
        )

    else:

        return (
            "Critical pollution level. Immediate pollution-control "
            "measures and continuous monitoring are recommended."
        )


# --------------------------------------------------
# POLLUTANT-SPECIFIC ACTION
# --------------------------------------------------

def pollutant_action(pollutant):

    if pollutant == "PM2.5":
        return (
            "Control combustion sources and fine-particle emissions."
        )

    elif pollutant == "PM10":
        return (
            "Focus on road dust and construction dust control."
        )

    elif pollutant == "NO2":
        return (
            "Focus on vehicle emissions and traffic management."
        )

    elif pollutant == "SO2":
        return (
            "Inspect fuel-burning and industrial emission sources."
        )

    elif pollutant == "Ozone":
        return (
            "Monitor ozone-forming pollutants and traffic emissions."
        )

    elif pollutant == "CO":
        return (
            "Monitor incomplete combustion and vehicle emissions."
        )

    return "Continue pollution monitoring."


# --------------------------------------------------
# GENERATE RECOMMENDATIONS
# --------------------------------------------------

df["AQI_Recommendation"] = df["AQI"].apply(
    get_aqi_recommendation
)

df["Pollutant_Action"] = df[
    "Dominant_Pollutant"
].apply(
    pollutant_action
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

df.to_csv(
    "data/processed/recommendations.csv",
    index=False
)


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print("\nAIR QUALITY RECOMMENDATIONS:\n")

for _, row in df.iterrows():

    print("--------------------------------")

    print("Location:", row["location"])
    print("AQI:", row["AQI"])

    print(
        "Dominant Pollutant:",
        row["Dominant_Pollutant"]
    )

    print(
        "Recommendation:",
        row["AQI_Recommendation"]
    )

    print(
        "Pollutant Action:",
        row["Pollutant_Action"]
    )


print(
    "\nSaved to:"
    " data/processed/recommendations.csv"
)