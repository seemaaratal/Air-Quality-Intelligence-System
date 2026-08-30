import folium
import plotly.express as px
import streamlit as st

from streamlit_folium import st_folium

from src.city_search import search_city
from src.data_source_manager import get_best_air_quality
from src.multi_city_live_prediction import predict_multi_city_aqi
from src.dynamic_hotspot_detection import scan_city_hotspots
from src.hotspot_recommendations import generate_hotspot_recommendations


# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="Air Quality Intelligence System",
    page_icon="🌍",
    layout="wide"
)


# ==================================================
# CACHE FUNCTIONS
# ==================================================

@st.cache_data(ttl=86400)
def cached_city_search(query):

    return search_city(
        query,
        count=5
    )


@st.cache_data(ttl=900)
def cached_best_air_quality(
    latitude,
    longitude
):

    return get_best_air_quality(
        latitude,
        longitude
    )


@st.cache_data(ttl=900)
def cached_multi_city_prediction(
    latitude,
    longitude,
    reference_time
):

    return predict_multi_city_aqi(
        latitude,
        longitude,
        reference_time
    )


@st.cache_data(ttl=900)
def cached_hotspot_scan(
    latitude,
    longitude,
    city_name
):

    return scan_city_hotspots(
        latitude,
        longitude,
        city_name
    )


# ==================================================
# AQI ZONE
# ==================================================

def get_zone(aqi):

    if aqi <= 100:
        return "Normal"

    elif aqi <= 200:
        return "Pollution Hotspot"

    elif aqi <= 300:
        return "Red Zone"

    else:
        return "Critical Red Zone"


# ==================================================
# MAP MARKER COLOR
# ==================================================

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


# ==================================================
# GENERAL RECOMMENDATION
# ==================================================

def show_general_recommendation(aqi):

    if aqi <= 50:

        return (
            "Air quality is good. Maintain regular monitoring, "
            "green cover and clean surroundings."
        )

    elif aqi <= 100:

        return (
            "Continue pollution monitoring. Encourage public "
            "transport and roadside greenery."
        )

    elif aqi <= 200:

        return (
            "Pollution hotspot requires attention. Monitor "
            "traffic, road dust and open waste burning."
        )

    elif aqi <= 300:

        return (
            "Red Zone detected. Increase traffic control "
            "and construction-dust management."
        )

    elif aqi <= 400:

        return (
            "Very poor air quality. Strict pollution-control "
            "measures and continuous monitoring are required."
        )

    else:

        return (
            "Severe pollution level. Immediate pollution-control "
            "action and continuous monitoring are required."
        )


# ==================================================
# POLLUTANT RECOMMENDATION
# ==================================================

def show_pollutant_action(pollutant):

    actions = {

        "PM2.5":
            (
                "Reduce combustion-related emissions, vehicle "
                "exhaust and fine-particle sources."
            ),

        "PM10":
            (
                "Control road dust and construction-related dust."
            ),

        "NO2":
            (
                "Reduce vehicle emissions and improve "
                "traffic management."
            ),

        "SO2":
            (
                "Monitor industrial and fuel-burning emissions."
            ),

        "Ozone":
            (
                "Monitor ozone-forming pollutants and "
                "traffic-related emissions."
            ),

        "CO":
            (
                "Reduce incomplete combustion and "
                "vehicle-related emissions."
            )
    }

    return actions.get(
        pollutant,
        "Continue regular pollution monitoring."
    )


# ==================================================
# HEADER
# ==================================================

st.title(
    "🌍 AI-Powered Air Quality Intelligence System"
)

st.write(
    "Search any city to view current air quality, "
    "24h/48h AI forecasts, pollution hotspots, "
    "red zones, plantation priorities and "
    "intelligent recommendations."
)

st.divider()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header(
    "📍 Location"
)


if "city_query" not in st.session_state:

    st.session_state.city_query = "Belagavi"


def clear_city():

    st.session_state.city_query = ""


city_query = st.sidebar.text_input(
    "Search any city",
    key="city_query",
    placeholder="Example: Delhi, Terdal, Hubballi"
)


st.sidebar.button(
    "🗑️ Clear Search",
    on_click=clear_city
)


if not city_query.strip():

    st.info(
        "👈 Enter a city name in the sidebar."
    )

    st.stop()


# ==================================================
# CITY SEARCH
# ==================================================

try:

    matches = cached_city_search(
        city_query
    )

except Exception:

    st.error(
        "City search service is unavailable right now."
    )

    st.stop()


if not matches:

    st.error(
        "City not found. Try another city name."
    )

    st.stop()


options = {

    match["display_name"]:
        match

    for match in matches
}


selected_label = st.sidebar.selectbox(
    "Choose matching location",
    list(options.keys())
)


selected_city = options[
    selected_label
]


city_name = selected_city[
    "name"
]


latitude = selected_city[
    "latitude"
]


longitude = selected_city[
    "longitude"
]


# ==================================================
# CURRENT AIR QUALITY
# ==================================================

try:

    with st.spinner(
        "Checking current air-quality data..."
    ):

        live = cached_best_air_quality(
            latitude,
            longitude
        )

except Exception:

    st.error(
        "Current air-quality data is unavailable."
    )

    st.stop()


zone = get_zone(
    live["AQI"]
)


# ==================================================
# LOCATION
# ==================================================

st.subheader(
    f"📍 Air Quality Status - {selected_label}"
)


st.caption(
    f"Coordinates: "
    f"{latitude:.4f}, {longitude:.4f}"
)


# ==================================================
# DATA SOURCE
# ==================================================

if live["Source"] == "OpenAQ Ground Station":

    st.success(
        "📡 Data Source: Fresh physical "
        "ground-station measurements\n\n"
        f"Station: {live['Source_Detail']}"
    )

else:

    st.info(
        "📡 Data Source: Open-Meteo / CAMS fallback\n\n"
        f"Ground-station status: "
        f"{live['Source_Detail']}"
    )


# ==================================================
# CURRENT AQI
# ==================================================

col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "AQI",
    live["AQI"]
)


col2.metric(
    "Category",
    live["Category"]
)


col3.metric(
    "Dominant Pollutant",
    live["Dominant_Pollutant"]
)


col4.metric(
    "Zone Status",
    zone
)


# ==================================================
# POLLUTANTS
# ==================================================

st.divider()

st.subheader(
    "🌫 Pollutant Levels"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "PM2.5",
        live["PM2.5"]
    )

    st.metric(
        "NO₂",
        live["NO2"]
    )


with col2:

    st.metric(
        "PM10",
        live["PM10"]
    )

    st.metric(
        "SO₂",
        live["SO2"]
    )


with col3:

    st.metric(
        "Ozone",
        live["Ozone"]
    )

    st.metric(
        "CO",
        live["CO"]
    )


st.caption(
    f"🕒 Latest data timestamp: "
    f"{live['Timestamp']}"
)


# ==================================================
# AI FORECAST
# ==================================================

st.divider()

st.subheader(
    "🔮 AI AQI Forecast"
)


try:

    with st.spinner(
        "Generating 24h and 48h AQI forecast..."
    ):

        forecast = cached_multi_city_prediction(
            latitude,
            longitude,
            str(live["Timestamp"])
        )


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Model Input AQI",
        forecast["Current_AQI"]
    )


    col2.metric(
        "24-Hour Forecast",
        forecast["Prediction_24h"]
    )

    col2.caption(
        forecast["Category_24h"]
    )


    col3.metric(
        "48-Hour Forecast",
        forecast["Prediction_48h"]
    )

    col3.caption(
        forecast["Category_48h"]
    )


    st.caption(
        f"Model: {forecast['Model']} | "
        f"Reference time: "
        f"{forecast['Reference_Time']}"
    )


    st.caption(
        f"24h target: "
        f"{forecast['Target_24h_Time']} | "
        f"48h target: "
        f"{forecast['Target_48h_Time']}"
    )


    st.info(
        "The forecasting model was trained on "
        "historical data from 10 representative "
        "Indian cities. Forecasts for other cities "
        "are generalized model predictions."
    )


except Exception:

    st.warning(
        "AI forecast is temporarily unavailable. "
        "Current AQI monitoring is still working."
    )


# ==================================================
# SMART RECOMMENDATIONS
# ==================================================

st.divider()

st.subheader(
    "🤖 Smart Recommendations"
)


aqi = live["AQI"]


if aqi <= 50:

    st.success(
        f"✅ Air quality in {city_name} is Good."
    )


elif aqi <= 100:

    st.info(
        f"ℹ️ Air quality in {city_name} "
        f"is Satisfactory."
    )


elif aqi <= 200:

    st.warning(
        f"⚠️ Pollution hotspot detected "
        f"in {city_name}."
    )


elif aqi <= 300:

    st.error(
        f"🚨 Red Zone detected "
        f"in {city_name}."
    )


else:

    st.error(
        f"🚨 Critical pollution level "
        f"in {city_name}!"
    )


st.write(
    "### 🌱 Recommended Action"
)


st.write(
    show_general_recommendation(
        aqi
    )
)


st.write(
    "### 🌫 Pollutant-Specific Action"
)


st.write(
    show_pollutant_action(
        live["Dominant_Pollutant"]
    )
)


# ==================================================
# HOTSPOT DETECTION
# ==================================================

st.divider()

st.subheader(
    "🔥 Dynamic Pollution Hotspot Detection"
)


st.caption(
    "The system scans multiple points around the "
    "selected city and classifies each location as "
    "Normal, Pollution Hotspot, Red Zone or "
    "Critical Red Zone."
)


try:

    with st.spinner(
        "Scanning pollution hotspots..."
    ):

        hotspot_df = cached_hotspot_scan(
            latitude,
            longitude,
            city_name
        )

except Exception:

    hotspot_df = None


# ==================================================
# HOTSPOT RESULTS
# ==================================================

if (
    hotspot_df is not None
    and not hotspot_df.empty
):

    # ----------------------------------------------
    # SUCCESSFUL SCAN STATUS
    # ----------------------------------------------

    expected_scans = 9
    successful_scans = len(hotspot_df)

    if successful_scans == expected_scans:

        st.success(
            f"✅ Successful scans: "
            f"{successful_scans}/{expected_scans}"
        )

    else:

        st.warning(
            f"⚠️ Successful scans: "
            f"{successful_scans}/{expected_scans}. "
            "Some scan points could not be retrieved "
            "because of API or network availability."
        )


    normal_count = (
        hotspot_df["Zone"]
        .eq("Normal")
        .sum()
    )


    hotspot_count = (
        hotspot_df["Zone"]
        .eq("Pollution Hotspot")
        .sum()
    )


    red_count = (
        hotspot_df["Zone"]
        .eq("Red Zone")
        .sum()
    )


    critical_count = (
        hotspot_df["Zone"]
        .eq("Critical Red Zone")
        .sum()
    )


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Normal Points",
        int(normal_count)
    )


    col2.metric(
        "Pollution Hotspots",
        int(hotspot_count)
    )


    col3.metric(
        "Red Zones",
        int(red_count)
    )


    col4.metric(
        "Critical Red Zones",
        int(critical_count)
    )


    # ----------------------------------------------
    # HIGHEST POLLUTION POINT
    # ----------------------------------------------

    worst_index = (
        hotspot_df["AQI"]
        .idxmax()
    )


    worst_area = hotspot_df.loc[
        worst_index
    ]


    st.write(
        "### 🚨 Highest Pollution Point"
    )


    st.write(
        f"**{worst_area['Area']}** — "
        f"AQI **{worst_area['AQI']}** | "
        f"{worst_area['Category']} | "
        f"{worst_area['Zone']} | "
        f"Dominant pollutant: "
        f"**{worst_area['Dominant_Pollutant']}**"
    )


    # ----------------------------------------------
    # HOTSPOT TABLE
    # ----------------------------------------------

    st.write(
        "### 📋 Hotspot Scan Results"
    )


    st.dataframe(
        hotspot_df[
            [
                "Area",
                "AQI",
                "Category",
                "Dominant_Pollutant",
                "Zone"
            ]
        ],
        width="stretch",
        hide_index=True
    )


    # ----------------------------------------------
    # HOTSPOT GRAPH
    # ----------------------------------------------

    st.write(
        "### 📊 Area-Wise AQI Comparison"
    )


    hotspot_fig = px.bar(
        hotspot_df,
        x="Area",
        y="AQI",
        color="Zone",
        text="AQI",
        title=(
            f"Pollution Hotspot Scan - {city_name}"
        )
    )


    st.plotly_chart(
        hotspot_fig,
        width="stretch"
    )


    st.caption(
        "Nearby points may show identical or similar "
        "AQI values because Open-Meteo/CAMS has "
        "coarser spatial resolution than street-level "
        "ground sensors."
    )


else:

    st.warning(
        "Hotspot scan is currently unavailable."
    )


# ==================================================
# PLANTATION + HOTSPOT RECOMMENDATIONS
# ==================================================

st.divider()

st.subheader(
    "🌱 Plantation & Hotspot Recommendations"
)


if (
    hotspot_df is not None
    and not hotspot_df.empty
):

    recommendation_df = (
        generate_hotspot_recommendations(
            hotspot_df
        )
    )


    if not recommendation_df.empty:

        low_count = (
            recommendation_df[
                "Plantation_Priority"
            ]
            .eq("Low")
            .sum()
        )


        moderate_count = (
            recommendation_df[
                "Plantation_Priority"
            ]
            .eq("Moderate")
            .sum()
        )


        high_count = (
            recommendation_df[
                "Plantation_Priority"
            ]
            .eq("High")
            .sum()
        )


        urgent_count = (
            recommendation_df[
                "Plantation_Priority"
            ]
            .eq("Urgent")
            .sum()
        )


        # ------------------------------------------
        # PRIORITY COUNTS
        # ------------------------------------------

        col1, col2, col3, col4 = st.columns(4)


        col1.metric(
            "Low Priority",
            int(low_count)
        )


        col2.metric(
            "Moderate Priority",
            int(moderate_count)
        )


        col3.metric(
            "High Priority",
            int(high_count)
        )


        col4.metric(
            "Urgent Priority",
            int(urgent_count)
        )


        # ------------------------------------------
        # PRIORITY TABLE
        # ------------------------------------------

        st.write(
            "### 🌳 Area-Wise Plantation Priority"
        )


        st.dataframe(
            recommendation_df[
                [
                    "Area",
                    "AQI",
                    "Zone",
                    "Dominant_Pollutant",
                    "Plantation_Priority"
                ]
            ],
            width="stretch",
            hide_index=True
        )


        # ------------------------------------------
        # PRIORITY GRAPH
        # ------------------------------------------

        priority_fig = px.bar(
            recommendation_df,
            x="Area",
            y="AQI",
            color="Plantation_Priority",
            text="Plantation_Priority",
            title=(
                f"Plantation Priority - {city_name}"
            )
        )


        st.plotly_chart(
            priority_fig,
            width="stretch"
        )


        # ------------------------------------------
        # MOST CRITICAL AREA RECOMMENDATION
        # ------------------------------------------

        recommendation_index = (
            recommendation_df["AQI"]
            .idxmax()
        )


        critical_area = (
            recommendation_df.loc[
                recommendation_index
            ]
        )


        st.write(
            "### 🎯 Priority Action"
        )


        st.write(
            f"**Area:** {critical_area['Area']}"
        )


        st.write(
            f"**Plantation Priority:** "
            f"{critical_area['Plantation_Priority']}"
        )


        st.write(
            f"**Zone Action:** "
            f"{critical_area['Zone_Action']}"
        )


        st.write(
            f"**Pollutant-Specific Action:** "
            f"{critical_area['Pollutant_Action']}"
        )


        # ------------------------------------------
        # DETAILED AREA RECOMMENDATIONS
        # ------------------------------------------

        with st.expander(
            "View recommendations for all scanned areas"
        ):

            for _, row in recommendation_df.iterrows():

                st.markdown(
                    f"#### 📍 {row['Area']}"
                )

                st.write(
                    f"**AQI:** {row['AQI']}"
                )

                st.write(
                    f"**Zone:** {row['Zone']}"
                )

                st.write(
                    f"**Dominant Pollutant:** "
                    f"{row['Dominant_Pollutant']}"
                )

                st.write(
                    f"**Plantation Priority:** "
                    f"{row['Plantation_Priority']}"
                )

                st.write(
                    f"**Zone Action:** "
                    f"{row['Zone_Action']}"
                )

                st.write(
                    f"**Pollutant Action:** "
                    f"{row['Pollutant_Action']}"
                )

                st.divider()


        st.caption(
            "Plantation recommendations are rule-based "
            "decision-support suggestions. Green buffers "
            "are treated as a supporting long-term measure "
            "and not as a replacement for emission control."
        )


    else:

        st.warning(
            "Recommendation generation is unavailable."
        )


else:

    st.warning(
        "Hotspot data is required to generate "
        "plantation recommendations."
    )


# ==================================================
# INTERACTIVE HOTSPOT MAP
# ==================================================

st.divider()

st.subheader(
    "🗺️ Interactive Pollution Hotspot Map"
)


m = folium.Map(
    location=[
        latitude,
        longitude
    ],
    zoom_start=11
)


if (
    hotspot_df is not None
    and not hotspot_df.empty
):

    for _, row in hotspot_df.iterrows():

        popup_text = f"""
        <b>{row['Area']}</b><br>
        AQI: {row['AQI']}<br>
        Category: {row['Category']}<br>
        Zone: {row['Zone']}<br>
        Dominant Pollutant:
        {row['Dominant_Pollutant']}
        """


        folium.Marker(

            location=[
                row["Latitude"],
                row["Longitude"]
            ],

            popup=folium.Popup(
                popup_text,
                max_width=280
            ),

            tooltip=(
                f"{row['Area']} - "
                f"AQI {row['AQI']} - "
                f"{row['Zone']}"
            ),

            icon=folium.Icon(
                color=get_marker_color(
                    row["AQI"]
                ),
                icon="info-sign"
            )

        ).add_to(m)


else:

    folium.Marker(
        location=[
            latitude,
            longitude
        ],
        tooltip=selected_label
    ).add_to(m)


st_folium(
    m,
    width="stretch",
    height=500
)


# ==================================================
# FINAL NOTE
# ==================================================

st.caption(
    "AQI values shown by the CAMS fallback are "
    "CPCB-style AQI values calculated by this "
    "prototype from modelled pollutant concentrations. "
    "They should not be interpreted as official CPCB "
    "station AQI readings."
)