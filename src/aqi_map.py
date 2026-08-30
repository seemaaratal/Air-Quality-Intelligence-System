import pandas as pd
import folium
import os


# Load red-zone results
df = pd.read_csv(
    "data/processed/red_zone_results.csv"
)


# ------------------------------------------
# MAP CENTER
# ------------------------------------------

map_center = [
    df["latitude"].mean(),
    df["longitude"].mean()
]

aqi_map = folium.Map(
    location=map_center,
    zoom_start=9
)


# ------------------------------------------
# MARKER COLOR
# ------------------------------------------

def get_marker_color(aqi):

    if aqi <= 50:
        return "green"

    elif aqi <= 100:
        return "blue"

    elif aqi <= 200:
        return "orange"

    elif aqi <= 300:
        return "red"

    else:
        return "darkred"


# ------------------------------------------
# ADD LOCATIONS TO MAP
# ------------------------------------------

for _, row in df.iterrows():

    popup_text = f"""
    <b>{row['location']}</b><br>
    AQI: {row['AQI']}<br>
    Category: {row['AQI_Category']}<br>
    Zone: {row['Zone_Status']}
    """

    folium.Marker(
        location=[
            row["latitude"],
            row["longitude"]
        ],

        popup=folium.Popup(
            popup_text,
            max_width=250
        ),

        tooltip=f"{row['location']} - AQI {row['AQI']}",

        icon=folium.Icon(
            color=get_marker_color(
                row["AQI"]
            ),
            icon="info-sign"
        )

    ).add_to(aqi_map)


# ------------------------------------------
# SAVE MAP
# ------------------------------------------

os.makedirs(
    "outputs/maps",
    exist_ok=True
)

output_file = (
    "outputs/maps/"
    "belagavi_aqi_map.html"
)

aqi_map.save(output_file)


print("\nAQI map generated successfully!")

print(
    "Map saved to:",
    output_file
)