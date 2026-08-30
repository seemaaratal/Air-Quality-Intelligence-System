import pandas as pd

from src.city_search import search_city
from src.dynamic_hotspot_detection import scan_city_hotspots


# --------------------------------------------------
# PLANTATION PRIORITY
# --------------------------------------------------

def get_plantation_priority(zone):

    if zone == "Normal":
        return "Low"

    elif zone == "Pollution Hotspot":
        return "Moderate"

    elif zone == "Red Zone":
        return "High"

    else:
        return "Urgent"


# --------------------------------------------------
# ZONE-BASED ACTION
# --------------------------------------------------

def get_zone_action(zone):

    if zone == "Normal":

        return (
            "Maintain existing green cover and "
            "continue regular pollution monitoring."
        )

    elif zone == "Pollution Hotspot":

        return (
            "Increase roadside plantation and green buffers. "
            "Control local traffic, road dust and waste burning."
        )

    elif zone == "Red Zone":

        return (
            "High-priority plantation and green-buffer development "
            "are recommended. Strengthen traffic and dust-control measures."
        )

    else:

        return (
            "Immediate pollution-control measures are required. "
            "Develop dense green buffers as a long-term supporting measure "
            "and continuously monitor pollution levels."
        )


# --------------------------------------------------
# POLLUTANT-SPECIFIC ACTION
# --------------------------------------------------

def get_pollutant_action(pollutant):

    actions = {

        "PM2.5":
            (
                "Reduce vehicle exhaust, combustion emissions "
                "and other fine-particle sources."
            ),

        "PM10":
            (
                "Control road and construction dust. "
                "Use roadside vegetation and green barriers "
                "where suitable."
            ),

        "NO2":
            (
                "Improve traffic flow, reduce vehicle idling "
                "and encourage public transport."
            ),

        "SO2":
            (
                "Monitor fuel-burning and industrial emission sources "
                "and strengthen green buffers near emission areas."
            ),

        "Ozone":
            (
                "Monitor traffic-related ozone-forming pollutants "
                "and reduce precursor emissions."
            ),

        "CO":
            (
                "Reduce incomplete combustion and vehicle emissions "
                "and improve traffic management."
            )
    }

    return actions.get(
        pollutant,
        "Continue regular pollution monitoring."
    )


# --------------------------------------------------
# GENERATE RECOMMENDATIONS
# --------------------------------------------------

def generate_hotspot_recommendations(hotspot_df):

    if hotspot_df is None or hotspot_df.empty:

        return pd.DataFrame()


    results = []


    for _, row in hotspot_df.iterrows():

        zone = row["Zone"]

        pollutant = row[
            "Dominant_Pollutant"
        ]


        results.append({

            "Area":
                row["Area"],

            "AQI":
                row["AQI"],

            "Zone":
                zone,

            "Dominant_Pollutant":
                pollutant,

            "Plantation_Priority":
                get_plantation_priority(
                    zone
                ),

            "Zone_Action":
                get_zone_action(
                    zone
                ),

            "Pollutant_Action":
                get_pollutant_action(
                    pollutant
                )
        })


    return pd.DataFrame(
        results
    )


# --------------------------------------------------
# TERMINAL TEST
# --------------------------------------------------

if __name__ == "__main__":

    city = input(
        "Enter city name: "
    )


    matches = search_city(
        city,
        count=5
    )


    if not matches:

        print(
            "City not found."
        )

        raise SystemExit


    print(
        "\nMatching locations:\n"
    )


    for index, item in enumerate(
        matches,
        start=1
    ):

        print(
            f"{index}. "
            f"{item['display_name']}"
        )


    choice = int(
        input(
            "\nSelect location number: "
        )
    )


    selected = matches[
        choice - 1
    ]


    hotspot_df = scan_city_hotspots(
        selected["latitude"],
        selected["longitude"],
        selected["name"]
    )


    recommendation_df = (
        generate_hotspot_recommendations(
            hotspot_df
        )
    )


    print(
        "\nHOTSPOT RECOMMENDATIONS\n"
    )


    print(
        recommendation_df[
            [
                "Area",
                "AQI",
                "Zone",
                "Dominant_Pollutant",
                "Plantation_Priority"
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\nDETAILED ACTIONS\n"
    )


    for _, row in recommendation_df.iterrows():

        print(
            f"\nArea: {row['Area']}"
        )

        print(
            f"AQI: {row['AQI']}"
        )

        print(
            f"Zone: {row['Zone']}"
        )

        print(
            "Plantation Priority:",
            row["Plantation_Priority"]
        )

        print(
            "Zone Action:",
            row["Zone_Action"]
        )

        print(
            "Pollutant Action:",
            row["Pollutant_Action"]
        )