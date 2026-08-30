import pandas as pd
import plotly.express as px
import streamlit as st


# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)


RESULT_FILE = "outputs/multi_city_model_results.csv"


# ==================================================
# LOAD RESULTS
# ==================================================

@st.cache_data
def load_results():

    return pd.read_csv(
        RESULT_FILE
    )


try:

    df = load_results()

except Exception:

    st.error(
        "Model evaluation results could not be loaded."
    )

    st.stop()


# ==================================================
# HEADER
# ==================================================

st.title(
    "📈 Multi-City LSTM Model Performance"
)

st.write(
    "This page compares the trained Multi-City LSTM "
    "with a persistence baseline for 24-hour and "
    "48-hour AQI forecasting."
)

st.info(
    "MAE = Mean Absolute Error. "
    "Lower MAE means the forecast is closer to "
    "the actual AQI values."
)

st.divider()


# ==================================================
# OVERALL PERFORMANCE
# ==================================================

st.subheader(
    "🎯 Overall Performance"
)


lstm_24 = df[
    "LSTM_MAE_24h"
].mean()

baseline_24 = df[
    "Baseline_MAE_24h"
].mean()

lstm_48 = df[
    "LSTM_MAE_48h"
].mean()

baseline_48 = df[
    "Baseline_MAE_48h"
].mean()


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "LSTM 24h MAE",
    f"{lstm_24:.2f}"
)


col2.metric(
    "Baseline 24h MAE",
    f"{baseline_24:.2f}"
)


col3.metric(
    "LSTM 48h MAE",
    f"{lstm_48:.2f}"
)


col4.metric(
    "Baseline 48h MAE",
    f"{baseline_48:.2f}"
)


# ==================================================
# MODEL COMPARISON MESSAGE
# ==================================================

if lstm_24 < baseline_24:

    st.success(
        "✅ For the 24-hour forecast, the LSTM has "
        "a lower average MAE than the persistence baseline."
    )

else:

    st.warning(
        "⚠️ For the 24-hour forecast, the persistence "
        "baseline has a slightly lower average MAE "
        "than the LSTM."
    )


if lstm_48 < baseline_48:

    st.success(
        "✅ For the 48-hour forecast, the LSTM has "
        "a slightly lower average MAE than the "
        "persistence baseline."
    )

else:

    st.warning(
        "⚠️ For the 48-hour forecast, the persistence "
        "baseline has a lower average MAE than the LSTM."
    )


st.caption(
    "The model and baseline are close overall. "
    "The results are presented without claiming "
    "that the LSTM strongly outperforms the baseline."
)


# ==================================================
# CITY-WISE TABLE
# ==================================================

st.divider()

st.subheader(
    "🏙️ City-Wise Model Results"
)


display_df = df.copy()


for column in [
    "LSTM_MAE_24h",
    "Baseline_MAE_24h",
    "LSTM_MAE_48h",
    "Baseline_MAE_48h"
]:

    display_df[column] = (
        display_df[column]
        .round(2)
    )


st.dataframe(
    display_df,
    width="stretch",
    hide_index=True
)


# ==================================================
# 24-HOUR COMPARISON
# ==================================================

st.divider()

st.subheader(
    "⏱️ 24-Hour Forecast Comparison"
)


chart_24 = df[
    [
        "City",
        "LSTM_MAE_24h",
        "Baseline_MAE_24h"
    ]
].melt(
    id_vars="City",
    var_name="Model",
    value_name="MAE"
)


chart_24["Model"] = (
    chart_24["Model"]
    .replace(
        {
            "LSTM_MAE_24h":
                "Multi-City LSTM",

            "Baseline_MAE_24h":
                "Persistence Baseline"
        }
    )
)


fig_24 = px.bar(
    chart_24,
    x="City",
    y="MAE",
    color="Model",
    barmode="group",
    title="24-Hour AQI Forecast MAE by City"
)


st.plotly_chart(
    fig_24,
    width="stretch"
)


# ==================================================
# 48-HOUR COMPARISON
# ==================================================

st.divider()

st.subheader(
    "🕒 48-Hour Forecast Comparison"
)


chart_48 = df[
    [
        "City",
        "LSTM_MAE_48h",
        "Baseline_MAE_48h"
    ]
].melt(
    id_vars="City",
    var_name="Model",
    value_name="MAE"
)


chart_48["Model"] = (
    chart_48["Model"]
    .replace(
        {
            "LSTM_MAE_48h":
                "Multi-City LSTM",

            "Baseline_MAE_48h":
                "Persistence Baseline"
        }
    )
)


fig_48 = px.bar(
    chart_48,
    x="City",
    y="MAE",
    color="Model",
    barmode="group",
    title="48-Hour AQI Forecast MAE by City"
)


st.plotly_chart(
    fig_48,
    width="stretch"
)


# ==================================================
# CITY WIN COUNTS
# ==================================================

st.divider()

st.subheader(
    "🏆 City-Level Comparison"
)


lstm_wins_24 = (
    df["LSTM_MAE_24h"]
    <
    df["Baseline_MAE_24h"]
).sum()


baseline_wins_24 = (
    df["Baseline_MAE_24h"]
    <
    df["LSTM_MAE_24h"]
).sum()


lstm_wins_48 = (
    df["LSTM_MAE_48h"]
    <
    df["Baseline_MAE_48h"]
).sum()


baseline_wins_48 = (
    df["Baseline_MAE_48h"]
    <
    df["LSTM_MAE_48h"]
).sum()


col1, col2 = st.columns(2)


with col1:

    st.write(
        "### 24-Hour"
    )

    st.metric(
        "Cities where LSTM performs better",
        int(lstm_wins_24)
    )

    st.metric(
        "Cities where baseline performs better",
        int(baseline_wins_24)
    )


with col2:

    st.write(
        "### 48-Hour"
    )

    st.metric(
        "Cities where LSTM performs better",
        int(lstm_wins_48)
    )

    st.metric(
        "Cities where baseline performs better",
        int(baseline_wins_48)
    )


# ==================================================
# BEST / HARDEST CITY
# ==================================================

st.divider()

st.subheader(
    "🔍 Performance Insights"
)


best_24 = df.loc[
    df["LSTM_MAE_24h"].idxmin()
]


hardest_24 = df.loc[
    df["LSTM_MAE_24h"].idxmax()
]


best_48 = df.loc[
    df["LSTM_MAE_48h"].idxmin()
]


hardest_48 = df.loc[
    df["LSTM_MAE_48h"].idxmax()
]


col1, col2 = st.columns(2)


with col1:

    st.write(
        "### ✅ Lowest LSTM Error"
    )

    st.write(
        f"**24h:** {best_24['City']} "
        f"— MAE {best_24['LSTM_MAE_24h']:.2f}"
    )

    st.write(
        f"**48h:** {best_48['City']} "
        f"— MAE {best_48['LSTM_MAE_48h']:.2f}"
    )


with col2:

    st.write(
        "### ⚠️ Highest LSTM Error"
    )

    st.write(
        f"**24h:** {hardest_24['City']} "
        f"— MAE {hardest_24['LSTM_MAE_24h']:.2f}"
    )

    st.write(
        f"**48h:** {hardest_48['City']} "
        f"— MAE {hardest_48['LSTM_MAE_48h']:.2f}"
    )


# ==================================================
# BASELINE EXPLANATION
# ==================================================

st.divider()

st.subheader(
    "📌 What is the Persistence Baseline?"
)


st.write(
    "The persistence baseline assumes that the future "
    "AQI will remain equal to the current AQI."
)


st.code(
    "Future AQI = Current AQI"
)


st.write(
    "Comparing the LSTM with this simple baseline helps "
    "check whether the AI model is learning useful "
    "temporal patterns instead of being evaluated alone."
)


# ==================================================
# MODEL INFORMATION
# ==================================================

st.divider()

st.subheader(
    "🧠 Model Information"
)


st.write(
    """
**Model:** Multi-City LSTM

**Training cities:** 10 representative Indian cities

**Input history:** Previous 72 hours

**Forecast horizons:** 24 hours and 48 hours

**Main inputs:** AQI, pollutant concentrations,
weather variables, time features, future weather
information and geographic coordinates.

**Output:** Future AQI predictions for +24h and +48h.
"""
)


# ==================================================
# LIMITATIONS
# ==================================================

with st.expander(
    "⚠️ Model Limitations",
    expanded=True
):

    st.markdown(
        """
- The model was trained using data from 10 Indian cities.
- Predictions for unseen cities are generalized prototype predictions.
- Training pollution data is mainly based on Open-Meteo/CAMS modelled concentrations.
- CAMS does not provide street-level pollution resolution.
- Forecast performance differs between cities.
- Delhi has much larger forecast errors than several other training cities.
- The LSTM does not strongly outperform the persistence baseline overall.
- Live prediction depends on the availability of pollution and weather APIs.
"""
    )


# ==================================================
# FINAL NOTE
# ==================================================

st.divider()

st.info(
    "This evaluation is presented as a prototype "
    "model-performance analysis. The system does not "
    "claim that its AQI predictions are official "
    "government forecasts."
)