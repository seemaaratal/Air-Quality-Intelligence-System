import time

import pandas as pd

from src.live_air_quality import get_live_air_quality
from src.city_search import search_city


def classify_zone(aqi):

    if aqi <= 100:
        return "Normal"

    elif aqi <= 200:
        return "Pollution Hotspot"

    elif aqi <= 300:
        return "Red Zone"

    else:
        return "Critical Red Zone"


def fetch_point_with_retry(
    latitude,
    longitude,
    attempts=3
):

    last_error = None

    for attempt in range(
        1,
        attempts + 1
    ):

        try:

            return get_live_air_quality(
                latitude,
                longitude
            )

        except Exception as error:

            last_error = error

            if attempt < attempts:

                print(
                    f"Attempt {attempt} failed. "
                    "Retrying..."
                )

                time.sleep(2)

    raise last_error


def scan_city_hotspots(
    latitude,
    longitude,
    city_name
):

    step = 0.03

    points = [
        (
            "Center",
            latitude,
            longitude
        ),

        (
            "North",
            latitude + step,
            longitude
        ),

        (
            "South",
            latitude - step,
            longitude
        ),

        (
            "East",
            latitude,
            longitude + step
        ),

        (
            "West",
            latitude,
            longitude - step
        ),

        (
            "North-East",
            latitude + step,
            longitude + step
        ),

        (
            "North-West",
            latitude + step,
            longitude - step
        ),

        (
            "South-East",
            latitude - step,
            longitude + step
        ),

        (
            "South-West",
            latitude - step,
            longitude - step
        )
    ]

    results = []


    for (
        area,
        lat,
        lon
    ) in points:

        try:

            air = fetch_point_with_retry(
                lat,
                lon,
                attempts=3
            )

            aqi = air[
                "AQI"
            ]

            results.append({

                "City":
                    city_name,

                "Area":
                    area,

                "Latitude":
                    lat,

                "Longitude":
                    lon,

                "AQI":
                    aqi,

                "Category":
                    air["Category"],

                "Dominant_Pollutant":
                    air[
                        "Dominant_Pollutant"
                    ],

                "Zone":
                    classify_zone(
                        aqi
                    )
            })


        except Exception as error:

            print(
                f"{area} failed after "
                f"3 attempts: {error}"
            )


        # Small delay between points
        time.sleep(0.5)


    return pd.DataFrame(
        results
    )


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


    print(
        f"\nScanning around "
        f"{selected['display_name']}..."
    )


    df = scan_city_hotspots(
        selected["latitude"],
        selected["longitude"],
        selected["name"]
    )


    print(
        "\nHOTSPOT RESULTS\n"
    )


    print(
        df[
            [
                "Area",
                "AQI",
                "Category",
                "Dominant_Pollutant",
                "Zone"
            ]
        ].to_string(
            index=False
        )
    )


    print(
        f"\nSuccessful points: "
        f"{len(df)}/9"
    )