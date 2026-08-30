import os
from datetime import datetime

import pandas as pd


REPORT_FILE = "data/processed/citizen_reports.csv"


# --------------------------------------------------
# CREATE REPORT FILE IF NOT PRESENT
# --------------------------------------------------

def initialize_report_file():

    folder = os.path.dirname(
        REPORT_FILE
    )

    os.makedirs(
        folder,
        exist_ok=True
    )

    if not os.path.exists(
        REPORT_FILE
    ):

        df = pd.DataFrame(
            columns=[
                "Timestamp",
                "City",
                "Area",
                "Latitude",
                "Longitude",
                "Issue_Type",
                "Description"
            ]
        )

        df.to_csv(
            REPORT_FILE,
            index=False
        )


# --------------------------------------------------
# SAVE CITIZEN REPORT
# --------------------------------------------------

def save_citizen_report(
    city,
    area,
    latitude,
    longitude,
    issue_type,
    description=""
):

    initialize_report_file()

    new_report = pd.DataFrame(
        [
            {
                "Timestamp":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "City":
                    city,

                "Area":
                    area,

                "Latitude":
                    latitude,

                "Longitude":
                    longitude,

                "Issue_Type":
                    issue_type,

                "Description":
                    description.strip()
            }
        ]
    )


    new_report.to_csv(
        REPORT_FILE,
        mode="a",
        header=False,
        index=False
    )


    return True


# --------------------------------------------------
# LOAD REPORTS
# --------------------------------------------------

def load_citizen_reports():

    initialize_report_file()

    try:

        df = pd.read_csv(
            REPORT_FILE
        )

        if df.empty:
            return df

        df = df.sort_values(
            "Timestamp",
            ascending=False
        )

        return df.reset_index(
            drop=True
        )

    except Exception:

        return pd.DataFrame()


# --------------------------------------------------
# GET REPORTS FOR ONE CITY
# --------------------------------------------------

def get_city_reports(city):

    df = load_citizen_reports()

    if df.empty:
        return df

    city_reports = df[
        df["City"]
        .astype(str)
        .str.lower()
        == city.lower()
    ]

    return city_reports.reset_index(
        drop=True
    )


# --------------------------------------------------
# TERMINAL TEST
# --------------------------------------------------

if __name__ == "__main__":

    print(
        "CITIZEN POLLUTION REPORT TEST\n"
    )


    city = input(
        "Enter city: "
    )


    area = input(
        "Enter area/locality: "
    )


    print(
        "\nIssue Types:"
    )

    issue_types = [
        "Heavy Traffic",
        "Road Dust",
        "Construction Dust",
        "Waste Burning",
        "Industrial Smoke",
        "Vehicle Smoke",
        "Other"
    ]


    for index, issue in enumerate(
        issue_types,
        start=1
    ):

        print(
            f"{index}. {issue}"
        )


    choice = int(
        input(
            "\nSelect issue number: "
        )
    )


    issue_type = issue_types[
        choice - 1
    ]


    description = input(
        "Enter short description: "
    )


    save_citizen_report(
        city=city,
        area=area,
        latitude=0.0,
        longitude=0.0,
        issue_type=issue_type,
        description=description
    )


    print(
        "\nCitizen report saved successfully."
    )


    print(
        "\nRECENT REPORTS\n"
    )


    reports = load_citizen_reports()


    if reports.empty:

        print(
            "No reports found."
        )

    else:

        print(
            reports.head(5)
            .to_string(
                index=False
            )
        )