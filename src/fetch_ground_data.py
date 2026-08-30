import os
from openaq import OpenAQ


API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    print("OpenAQ API key not found!")
    exit()


STATION_ID = 358665


with OpenAQ(api_key=API_KEY) as client:

    # Get sensor details
    sensors_response = client.locations.sensors(STATION_ID)

    # Get latest measurements
    latest_response = client.locations.latest(STATION_ID)

    sensors = sensors_response.results
    latest = latest_response.results


# Sensor ID -> parameter information
sensor_info = {}

for sensor in sensors:

    sensor_info[sensor.id] = {
        "name": sensor.parameter.name,
        "display_name": sensor.parameter.display_name,
        "unit": sensor.parameter.units
    }


print("\nRAMTEERTH NAGAR GROUND-STATION DATA\n")


for measurement in latest:

    sensor_id = measurement.sensors_id

    if sensor_id not in sensor_info:
        continue

    info = sensor_info[sensor_id]

    print("--------------------------------")

    print(
        "Parameter:",
        info["display_name"] or info["name"]
    )

    print(
        "Value:",
        measurement.value,
        info["unit"]
    )

    print(
        "Time:",
        measurement.datetime.local
    )