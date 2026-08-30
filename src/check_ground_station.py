import os
from openaq import OpenAQ


API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    print("OpenAQ API key not found!")
    print("Set OPENAQ_API_KEY first.")
    exit()


# Belagavi coordinates
latitude = 15.8521
longitude = 74.5045


print("Searching for ground monitoring stations near Belagavi...\n")


with OpenAQ(api_key=API_KEY) as client:

    response = client.locations.list(
        coordinates=(latitude, longitude),

        # Maximum supported radius = 25 km
        radius=25000,

        # Reference-grade physical monitors only
        monitor=True,

        mobile=False,

        limit=100
    )


    locations = response.results


    if len(locations) == 0:

        print("No OpenAQ ground station found within 25 km.")
        print("System will use Open-Meteo/CAMS fallback.")

    else:

        print(
            f"Found {len(locations)} ground station(s):\n"
        )

        for location in locations:

            print("-----------------------------")

            print("Station ID:", location.id)
            print("Name:", location.name)

            print(
                "Latitude:",
                location.coordinates.latitude
            )

            print(
                "Longitude:",
                location.coordinates.longitude
            )

            print(
                "Provider:",
                location.provider.name
            )

            print(
                "Last Update:",
                location.datetime_last.local
            )

            print("\nSensors:")

            for sensor in location.sensors:

                print(
                    "-",
                    sensor.parameter.display_name
                    or sensor.parameter.name,
                    "(",
                    sensor.parameter.units,
                    ")"
                )