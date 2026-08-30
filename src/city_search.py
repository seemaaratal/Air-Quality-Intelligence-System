import time
import requests


GEOCODING_URL = (
    "https://geocoding-api.open-meteo.com/v1/search"
)


def search_city(
    query,
    count=5,
    attempts=3
):

    query = query.strip()

    if not query:
        return []

    params = {
        "name": query,
        "count": count,
        "language": "en",
        "format": "json"
    }

    last_error = None


    for attempt in range(
        1,
        attempts + 1
    ):

        try:

            response = requests.get(
                GEOCODING_URL,
                params=params,
                timeout=(20, 60)
            )

            response.raise_for_status()

            data = response.json()

            results = []


            for item in data.get(
                "results",
                []
            ):

                name = item.get(
                    "name",
                    "Unknown"
                )

                admin1 = item.get(
                    "admin1"
                )

                country = item.get(
                    "country"
                )


                parts = [
                    name
                ]


                if (
                    admin1
                    and admin1 not in parts
                ):

                    parts.append(
                        admin1
                    )


                if (
                    country
                    and country not in parts
                ):

                    parts.append(
                        country
                    )


                results.append({

                    "name":
                        name,

                    "display_name":
                        ", ".join(parts),

                    "latitude":
                        float(
                            item["latitude"]
                        ),

                    "longitude":
                        float(
                            item["longitude"]
                        ),

                    "country":
                        country,

                    "country_code":
                        item.get(
                            "country_code"
                        ),

                    "timezone":
                        item.get(
                            "timezone"
                        )
                })


            return results


        except (
            requests.exceptions.RequestException
        ) as error:

            last_error = error

            print(
                f"City search attempt "
                f"{attempt} failed."
            )


            if attempt < attempts:

                print(
                    "Retrying city search..."
                )

                time.sleep(3)


    raise last_error


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    city = input(
        "Enter city name: "
    )


    try:

        matches = search_city(
            city
        )


        if not matches:

            print(
                "City not found."
            )


        else:

            print(
                "\nMatching cities:\n"
            )


            for index, match in enumerate(
                matches,
                start=1
            ):

                print(
                    index,
                    match["display_name"],
                    match["latitude"],
                    match["longitude"]
                )


    except Exception as error:

        print(
            "\nCity search failed:"
        )

        print(
            type(error).__name__,
            "-",
            error
        )