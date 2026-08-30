import os
import time

from datetime import (
    datetime,
    timedelta,
    timezone
)

from openaq import OpenAQ

from src.live_air_quality import (
    calculate_aqi,
    get_live_air_quality
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MAX_AGE_HOURS = 6

SEARCH_RADIUS_METERS = 25000

RETRY_ATTEMPTS = 3

RETRY_DELAY_SECONDS = 2

REQUIRED_POLLUTANTS = (
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "Ozone",
    "CO"
)


# --------------------------------------------------
# OPENAQ RETRY
# --------------------------------------------------

def _retry_openaq(
    label,
    function,
    attempts=RETRY_ATTEMPTS,
    **kwargs
):

    last_error = None

    for attempt in range(
        1,
        attempts + 1
    ):

        try:

            return function(
                **kwargs
            )

        except Exception as error:

            last_error = error

            print(
                f"[OpenAQ] {label} "
                f"attempt {attempt}/{attempts} "
                f"failed: "
                f"{type(error).__name__}"
            )

            if attempt < attempts:

                time.sleep(
                    RETRY_DELAY_SECONDS
                    * attempt
                )

    raise last_error


# --------------------------------------------------
# DATETIME CONVERSION
# --------------------------------------------------

def _parse_datetime(value):

    if value is None:

        raise ValueError(
            "Missing datetime value."
        )

    if isinstance(
        value,
        datetime
    ):

        dt = value

    else:

        dt = datetime.fromisoformat(
            str(value).replace(
                "Z",
                "+00:00"
            )
        )

    if dt.tzinfo is None:

        dt = dt.replace(
            tzinfo=timezone.utc
        )

    return dt.astimezone(
        timezone.utc
    )


# --------------------------------------------------
# NORMALIZE TEXT
# --------------------------------------------------

def _normalize_text(value):

    return (
        str(value or "")
        .lower()
        .replace("₂", "2")
        .replace("₃", "3")
        .replace(".", "")
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


# --------------------------------------------------
# CHECK MASS UNIT
# --------------------------------------------------

def _is_mass_unit(unit):

    normalized = (
        str(unit or "")
        .lower()
        .replace("μ", "µ")
        .replace("³", "3")
        .replace(" ", "")
    )

    return (
        "µg/m3" in normalized
        or
        "ug/m3" in normalized
    )


# --------------------------------------------------
# IDENTIFY POLLUTANT
# --------------------------------------------------

def _pollutant_name(sensor):

    name = _normalize_text(
        sensor.parameter.name
    )

    display = _normalize_text(
        sensor.parameter.display_name
    )

    text = (
        f"{name}{display}"
    )

    if "pm25" in text:
        return "PM2.5"

    if "pm10" in text:
        return "PM10"

    if (
        "no2" in text
        or
        "nitrogendioxide" in text
    ):
        return "NO2"

    if (
        "so2" in text
        or
        "sulphurdioxide" in text
        or
        "sulfurdioxide" in text
    ):
        return "SO2"

    if (
        "o3" in text
        or
        "ozone" in text
    ):
        return "Ozone"

    if (
        name == "co"
        or
        display.startswith("co")
        or
        "carbonmonoxide" in text
    ):
        return "CO"

    return None


# --------------------------------------------------
# SAFE SENSOR DATETIME
# --------------------------------------------------

def _sensor_last_time(sensor):

    try:

        return _parse_datetime(
            sensor.datetime_last.utc
        )

    except Exception:

        return datetime.min.replace(
            tzinfo=timezone.utc
        )


# --------------------------------------------------
# SELECT MASS-CONCENTRATION SENSORS
# --------------------------------------------------

def _select_mass_sensors(sensors):

    selected = {}

    for sensor in sensors:

        if not _is_mass_unit(
            sensor.parameter.units
        ):
            continue

        pollutant = _pollutant_name(
            sensor
        )

        if pollutant not in REQUIRED_POLLUTANTS:
            continue

        if pollutant not in selected:

            selected[
                pollutant
            ] = sensor

            continue

        old_time = _sensor_last_time(
            selected[pollutant]
        )

        new_time = _sensor_last_time(
            sensor
        )

        if new_time > old_time:

            selected[
                pollutant
            ] = sensor

    return selected


# --------------------------------------------------
# SENSOR FRESHNESS
# --------------------------------------------------

def _sensor_is_fresh(
    sensor,
    now
):

    try:

        last_time = _parse_datetime(
            sensor.datetime_last.utc
        )

    except Exception:

        return False

    age_hours = (
        now - last_time
    ).total_seconds() / 3600

    return (
        -1
        <= age_hours
        <= MAX_AGE_HOURS
    )


# --------------------------------------------------
# GET HOURLY AVERAGE
# --------------------------------------------------

def _hourly_average(
    client,
    sensor_id,
    hours,
    now
):

    start = (
        now
        - timedelta(
            hours=hours + 3
        )
    )

    try:

        response = _retry_openaq(

            label=(
                f"sensor {sensor_id} "
                f"{hours}h measurements"
            ),

            function=
                client.measurements.list,

            sensors_id=
                sensor_id,

            data=
                "hours",

            datetime_from=
                start,

            datetime_to=
                now,

            limit=
                1000
        )

    except Exception as error:

        print(
            f"[OpenAQ] Measurement fetch "
            f"failed after retries for "
            f"sensor {sensor_id}: "
            f"{type(error).__name__}"
        )

        return None


    values = []

    cutoff = (
        now
        - timedelta(
            hours=hours
        )
    )


    for measurement in response.results:

        try:

            measurement_time = (
                _parse_datetime(
                    measurement
                    .period
                    .datetime_from
                    .utc
                )
            )

        except Exception:

            continue


        if (
            measurement_time < cutoff
            or
            measurement_time > now
        ):
            continue


        if measurement.value is not None:

            try:

                values.append(
                    float(
                        measurement.value
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                continue


    # Prototype coverage:
    # 24h -> at least 18 hourly readings
    # 8h  -> at least 6 hourly readings

    minimum_required = (
        18
        if hours == 24
        else 6
    )


    if len(values) < minimum_required:

        return None


    return (
        sum(values)
        / len(values)
    )


# --------------------------------------------------
# CAMS FALLBACK
# --------------------------------------------------

def _fallback(
    latitude,
    longitude,
    reason
):

    data = get_live_air_quality(
        latitude,
        longitude
    )

    data[
        "Source"
    ] = "Open-Meteo / CAMS"

    data[
        "Source_Detail"
    ] = reason

    data[
        "Station_Name"
    ] = None

    data[
        "Station_ID"
    ] = None

    return data


# --------------------------------------------------
# BEST DATA SOURCE
# --------------------------------------------------

def get_best_air_quality(
    latitude,
    longitude
):

    # Read the key when the function runs.
    # This avoids freezing an old/missing key
    # when the module is first imported.

    api_key = os.getenv(
        "OPENAQ_API_KEY"
    )


    # ----------------------------------------------
    # API KEY NOT AVAILABLE
    # ----------------------------------------------

    if not api_key:

        return _fallback(
            latitude,
            longitude,
            "OpenAQ API key unavailable."
        )


    now = datetime.now(
        timezone.utc
    )


    try:

        with OpenAQ(
            api_key=api_key
        ) as client:


            # --------------------------------------
            # SEARCH NEARBY PHYSICAL STATIONS
            # --------------------------------------

            try:

                locations_response = (
                    _retry_openaq(

                        label=
                            "nearby station search",

                        function=
                            client.locations.list,

                        coordinates=(
                            latitude,
                            longitude
                        ),

                        radius=
                            SEARCH_RADIUS_METERS,

                        monitor=
                            True,

                        mobile=
                            False,

                        limit=
                            20
                    )
                )

            except Exception as error:

                return _fallback(

                    latitude,
                    longitude,

                    (
                        "OpenAQ nearby station "
                        "search failed after retries: "
                        f"{type(error).__name__}."
                    )
                )


            locations = list(
                locations_response.results
            )


            if not locations:

                return _fallback(

                    latitude,
                    longitude,

                    (
                        "No reference-grade "
                        "ground station found "
                        "within 25 km."
                    )
                )


            # --------------------------------------
            # NEAREST FIRST
            # --------------------------------------

            locations.sort(

                key=lambda location: (

                    location.distance

                    if location.distance
                    is not None

                    else float("inf")
                )
            )


            station_reasons = []


            # --------------------------------------
            # CHECK NEAREST 5 STATIONS
            # --------------------------------------

            for location in locations[:5]:


                # ----------------------------------
                # SENSOR LIST WITH RETRY
                # ----------------------------------

                try:

                    sensors_response = (
                        _retry_openaq(

                            label=(
                                f"{location.name} "
                                f"sensor search"
                            ),

                            function=
                                client.locations.sensors,

                            locations_id=
                                location.id
                        )
                    )

                except TypeError:

                    # Some OpenAQ SDK versions accept
                    # the location ID positionally.

                    try:

                        sensors_response = (
                            _retry_openaq(

                                label=(
                                    f"{location.name} "
                                    f"sensor search"
                                ),

                                function=lambda:
                                    client
                                    .locations
                                    .sensors(
                                        location.id
                                    )
                            )
                        )

                    except Exception as error:

                        station_reasons.append(

                            f"{location.name}: "
                            f"sensor lookup failed "
                            f"({type(error).__name__})"
                        )

                        continue


                except Exception as error:

                    station_reasons.append(

                        f"{location.name}: "
                        f"sensor lookup failed "
                        f"({type(error).__name__})"
                    )

                    continue


                sensors = (
                    _select_mass_sensors(
                        sensors_response.results
                    )
                )


                # ----------------------------------
                # CHECK REQUIRED POLLUTANTS
                # ----------------------------------

                missing = [

                    pollutant

                    for pollutant
                    in REQUIRED_POLLUTANTS

                    if pollutant
                    not in sensors
                ]


                if missing:

                    station_reasons.append(

                        f"{location.name}: "
                        f"missing "
                        + ", ".join(
                            missing
                        )
                    )

                    continue


                # ----------------------------------
                # CHECK SENSOR FRESHNESS
                # ----------------------------------

                stale = [

                    pollutant

                    for pollutant,
                    sensor
                    in sensors.items()

                    if not _sensor_is_fresh(
                        sensor,
                        now
                    )
                ]


                if stale:

                    station_reasons.append(

                        f"{location.name}: "
                        f"stale "
                        + ", ".join(
                            stale
                        )
                    )

                    # IMPORTANT:
                    # Do not request hourly data
                    # when the sensor itself is stale.

                    continue


                # ----------------------------------
                # 24-HOUR POLLUTANTS
                # ----------------------------------

                pm25 = _hourly_average(
                    client,
                    sensors["PM2.5"].id,
                    24,
                    now
                )


                pm10 = _hourly_average(
                    client,
                    sensors["PM10"].id,
                    24,
                    now
                )


                no2 = _hourly_average(
                    client,
                    sensors["NO2"].id,
                    24,
                    now
                )


                so2 = _hourly_average(
                    client,
                    sensors["SO2"].id,
                    24,
                    now
                )


                # ----------------------------------
                # 8-HOUR POLLUTANTS
                # ----------------------------------

                ozone = _hourly_average(
                    client,
                    sensors["Ozone"].id,
                    8,
                    now
                )


                co_ug = _hourly_average(
                    client,
                    sensors["CO"].id,
                    8,
                    now
                )


                averages = [
                    pm25,
                    pm10,
                    no2,
                    so2,
                    ozone,
                    co_ug
                ]


                if any(
                    value is None
                    for value in averages
                ):

                    station_reasons.append(

                        f"{location.name}: "
                        f"insufficient recent "
                        f"hourly data or "
                        f"measurement request timeout"
                    )

                    continue


                # ----------------------------------
                # CO µg/m³ -> mg/m³
                # ----------------------------------

                co = (
                    co_ug / 1000
                )


                # ----------------------------------
                # AQI CALCULATION
                # ----------------------------------

                (
                    aqi,
                    category,
                    dominant

                ) = calculate_aqi(
                    pm25,
                    pm10,
                    no2,
                    so2,
                    ozone,
                    co
                )


                latest_used_time = max(

                    _sensor_last_time(
                        sensor
                    )

                    for sensor
                    in sensors.values()
                )


                provider_name = (

                    location.provider.name

                    if location.provider
                    is not None

                    else "OpenAQ provider"
                )


                # ----------------------------------
                # GROUND STATION RESULT
                # ----------------------------------

                return {

                    "AQI":
                        aqi,

                    "Category":
                        category,

                    "Dominant_Pollutant":
                        dominant,

                    "PM2.5":
                        round(
                            pm25,
                            2
                        ),

                    "PM10":
                        round(
                            pm10,
                            2
                        ),

                    "NO2":
                        round(
                            no2,
                            2
                        ),

                    "SO2":
                        round(
                            so2,
                            2
                        ),

                    "Ozone":
                        round(
                            ozone,
                            2
                        ),

                    "CO":
                        round(
                            co,
                            3
                        ),

                    "Timestamp":
                        latest_used_time
                        .astimezone()
                        .strftime(
                            "%Y-%m-%d "
                            "%H:%M:%S %Z"
                        ),

                    "Source":
                        "OpenAQ Ground Station",

                    "Source_Detail":
                        (
                            f"{location.name} "
                            f"- {provider_name}"
                        ),

                    "Station_Name":
                        location.name,

                    "Station_ID":
                        location.id
                }


            # --------------------------------------
            # NO SUITABLE STATION
            # --------------------------------------

            reason = (
                "Nearby ground station data "
                "is unavailable, stale, "
                "incomplete, or lacks enough "
                "hourly coverage."
            )


            if station_reasons:

                reason += (
                    " "
                    + station_reasons[0]
                )


            return _fallback(
                latitude,
                longitude,
                reason
            )


    except Exception as error:

        return _fallback(

            latitude,
            longitude,

            (
                "Ground-station check failed "
                "after retries: "
                f"{type(error).__name__}."
            )
        )


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    result = get_best_air_quality(
        15.8521,
        74.5045
    )


    print(
        "\nAIR QUALITY SOURCE CHECK\n"
    )


    for key, value in result.items():

        print(
            f"{key}: {value}"
        )