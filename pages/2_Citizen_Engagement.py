import plotly.express as px
import streamlit as st

from src.city_search import search_city
from src.citizen_reports import (
    save_citizen_report,
    get_city_reports
)


st.set_page_config(
    page_title="Citizen Engagement",
    page_icon="📢",
    layout="wide"
)


@st.cache_data(ttl=86400)
def cached_city_search(query):
    return search_city(
        query,
        count=5
    )


st.title(
    "📢 Citizen Engagement"
)

st.write(
    "Citizens can report local pollution issues such as "
    "road dust, traffic, waste burning or industrial smoke."
)

st.divider()


st.subheader(
    "📍 Select Location"
)


city_query = st.text_input(
    "Enter city name",
    value="Belagavi",
    placeholder="Example: Belagavi, Delhi, Hubballi"
)


if not city_query.strip():
    st.info(
        "Enter a city name."
    )
    st.stop()


try:
    matches = cached_city_search(
        city_query
    )

except Exception:
    st.error(
        "City search service is temporarily unavailable."
    )
    st.stop()


if not matches:
    st.warning(
        "City not found."
    )
    st.stop()


options = {
    match["display_name"]: match
    for match in matches
}


selected_label = st.selectbox(
    "Choose matching location",
    list(options.keys())
)


selected = options[
    selected_label
]


city_name = selected[
    "name"
]

latitude = selected[
    "latitude"
]

longitude = selected[
    "longitude"
]


st.caption(
    f"Selected: {selected_label} | "
    f"{latitude:.4f}, {longitude:.4f}"
)


st.divider()

st.subheader(
    "📝 Report a Pollution Issue"
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


with st.form(
    "citizen_report_form",
    clear_on_submit=True
):

    area = st.text_input(
        "Area / Locality",
        placeholder="Example: College Road"
    )

    issue_type = st.selectbox(
        "Issue Type",
        issue_types
    )

    description = st.text_area(
        "Short Description",
        placeholder=(
            "Example: Heavy road dust observed "
            "near the junction."
        ),
        max_chars=300
    )

    submitted = st.form_submit_button(
        "📨 Submit Report"
    )


if submitted:

    if not area.strip():

        st.warning(
            "Please enter the area/locality."
        )

    else:

        try:

            save_citizen_report(
                city=city_name,
                area=area.strip(),
                latitude=latitude,
                longitude=longitude,
                issue_type=issue_type,
                description=description
            )

            st.success(
                "✅ Pollution report submitted successfully."
            )

        except Exception:
            st.error(
                "Unable to save the report."
            )


st.caption(
    "The saved coordinates represent the selected city "
    "reference location, not the exact street-level "
    "location of the reported issue."
)


st.divider()

st.subheader(
    f"📊 Citizen Reports - {city_name}"
)


reports = get_city_reports(
    city_name
)


if reports.empty:

    st.info(
        "No citizen reports have been submitted "
        "for this city yet."
    )


else:

    total_reports = len(
        reports
    )

    most_common_issue = (
        reports["Issue_Type"]
        .mode()
        .iloc[0]
    )

    unique_areas = (
        reports["Area"]
        .nunique()
    )


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Total Reports",
        total_reports
    )

    col2.metric(
        "Reported Areas",
        unique_areas
    )

    col3.metric(
        "Most Common Issue",
        most_common_issue
    )


    st.write(
        "### 📋 Recent Reports"
    )


    st.dataframe(
        reports[
            [
                "Timestamp",
                "Area",
                "Issue_Type",
                "Description"
            ]
        ],
        width="stretch",
        hide_index=True
    )


    issue_counts = (
        reports["Issue_Type"]
        .value_counts()
        .reset_index()
    )

    issue_counts.columns = [
        "Issue",
        "Reports"
    ]


    st.write(
        "### 📈 Reported Pollution Issues"
    )


    fig = px.bar(
        issue_counts,
        x="Issue",
        y="Reports",
        text="Reports",
        title=(
            f"Citizen Pollution Reports - {city_name}"
        )
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


st.divider()

st.info(
    "Citizen reports are supplementary observations. "
    "They are not treated as verified sensor measurements "
    "and do not directly change the calculated AQI."
)