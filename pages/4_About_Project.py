import streamlit as st


# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="About Project",
    page_icon="ℹ️",
    layout="wide"
)


# ==================================================
# HEADER
# ==================================================

st.title("ℹ️ About the Project")

st.subheader(
    "AI-Powered Spatio-Temporal Air Quality Intelligence System"
)

st.write(
    """
This project is an air-quality decision-support prototype
that combines pollution data, weather information,
machine-learning forecasting, spatial super-resolution,
hotspot detection and citizen reporting in one
interactive system.
"""
)

st.divider()


# ==================================================
# PROBLEM
# ==================================================

st.subheader("🎯 Problem Statement")

st.write(
    """
Traditional air-quality monitoring depends heavily on
ground stations. These stations may be sparse, unavailable
or incomplete in many locations.

Atmospheric pollution datasets can also have coarse spatial
resolution, making detailed spatial analysis difficult.

This project combines available ground-station data with
modelled atmospheric data, AI forecasting and an
experimental spatial super-resolution module to provide
wider air-quality coverage and intelligent analysis.
"""
)


# ==================================================
# OBJECTIVES
# ==================================================

st.divider()

st.subheader("✅ Project Objectives")

st.markdown(
    """
- Monitor current air quality for searched cities.
- Calculate CPCB-style AQI from pollutant concentrations.
- Forecast AQI for the next 24 and 48 hours.
- Detect pollution hotspots and red zones.
- Identify dominant pollutants.
- Generate pollution-control recommendations.
- Suggest plantation priority for polluted areas.
- Support citizen reporting of local pollution issues.
- Experimentally enhance coarse AQI spatial surfaces using an SRGAN-style model.
- Display results using interactive dashboards, maps and model-analysis pages.
"""
)


# ==================================================
# SYSTEM WORKFLOW
# ==================================================

st.divider()

st.subheader("⚙️ System Workflow")

st.code(
    """
City Search
     ↓
Latitude & Longitude
     ↓
Ground Station Check (OpenAQ)
     ↓
Fresh & Complete?
   ↙       ↘
 Yes       No
  ↓         ↓
Ground     Open-Meteo /
Data       CAMS Fallback
   ↘       ↙
     AQI Calculation
           ↓
    Current AQI Status
           ↓
 ┌─────────┼───────────┐
 ↓         ↓           ↓
LSTM     Hotspot     Smart
Forecast Detection   Recommendations
 ↓         ↓           ↓
24h/48h  Red Zones   Plantation
Forecast              Priority
           ↓
     Streamlit Dashboard


Separate Spatial Enhancement Experiment:

10-City AQI Time Series
           ↓
Inverse-Distance Interpolation
           ↓
Target AQI Surface (32 × 32)
           ↓
Downsampling (8 × 8)
           ↓
Bicubic + Residual Generator
           ↓
Adversarial Training
           ↓
SRGAN-Enhanced AQI Surface (32 × 32)
"""
)


# ==================================================
# MAIN MODULES
# ==================================================

st.divider()

st.subheader("🧩 Main System Modules")

col1, col2 = st.columns(2)


with col1:

    st.write("### 📍 1. City Search")

    st.write(
        "Searches a city and retrieves its "
        "latitude and longitude."
    )


    st.write("### 📡 2. Data Source Selection")

    st.write(
        "Checks OpenAQ ground stations first. "
        "If suitable fresh data is unavailable, "
        "the system uses Open-Meteo/CAMS data."
    )


    st.write("### 🌫 3. AQI Calculation")

    st.write(
        "Uses PM2.5, PM10, NO₂, SO₂, Ozone "
        "and CO concentrations to calculate "
        "a CPCB-style AQI."
    )


    st.write("### 🔮 4. AQI Forecasting")

    st.write(
        "A Multi-City LSTM uses the previous "
        "72 hours of pollution and weather data "
        "to generate 24h and 48h AQI forecasts."
    )


    st.write("### 🗺️ 5. Spatial Super-Resolution")

    st.write(
        "An experimental SRGAN-style module enhances "
        "8 × 8 coarse AQI surfaces to 32 × 32 spatial "
        "surfaces using residual learning and "
        "adversarial training."
    )


with col2:

    st.write("### 🔥 6. Hotspot Detection")

    st.write(
        "Scans multiple geographic points around "
        "the selected city and classifies them as "
        "Normal, Pollution Hotspot, Red Zone or "
        "Critical Red Zone."
    )


    st.write("### 🌱 7. Plantation Recommendations")

    st.write(
        "Assigns Low, Moderate, High or Urgent "
        "plantation priority using AQI zone and "
        "dominant pollutant information."
    )


    st.write("### 🤖 8. Smart Recommendations")

    st.write(
        "Provides rule-based pollution-control "
        "suggestions according to AQI and "
        "dominant pollutant."
    )


    st.write("### 📢 9. Citizen Engagement")

    st.write(
        "Allows citizens to report local pollution "
        "issues such as road dust, traffic, waste "
        "burning and industrial smoke."
    )


# ==================================================
# AI FORECASTING MODEL
# ==================================================

st.divider()

st.subheader("🧠 AI Forecasting Model")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Training Cities",
    "10"
)

col2.metric(
    "Input History",
    "72 Hours"
)

col3.metric(
    "Forecast",
    "24h + 48h"
)

st.markdown(
    """
The Multi-City LSTM receives:

- Historical AQI
- PM2.5
- PM10
- CO
- NO₂
- SO₂
- Ozone
- Temperature
- Relative humidity
- Precipitation
- Wind speed
- Time features
- Future weather information
- Latitude and longitude

The model predicts AQI changes for the next
24 and 48 hours.
"""
)


# ==================================================
# LSTM MODEL EVALUATION
# ==================================================

st.divider()

st.subheader("📈 LSTM Model Evaluation")

st.write(
    """
The LSTM was evaluated against a persistence baseline.

A persistence baseline assumes:
"""
)

st.code(
    "Future AQI = Current AQI"
)


col1, col2 = st.columns(2)


with col1:

    st.write("### 24-Hour Forecast")

    st.write(
        "**LSTM MAE:** 12.39"
    )

    st.write(
        "**Baseline MAE:** 12.30"
    )

    st.caption(
        "Baseline performs slightly better."
    )


with col2:

    st.write("### 48-Hour Forecast")

    st.write(
        "**LSTM MAE:** 16.66"
    )

    st.write(
        "**Baseline MAE:** 16.70"
    )

    st.caption(
        "LSTM performs slightly better."
    )


st.info(
    "The results are close overall. "
    "The project does not claim that the LSTM "
    "strongly outperforms the persistence baseline."
)


# ==================================================
# SRGAN MODEL
# ==================================================

st.divider()

st.subheader("🗺️ Spatial Super-Resolution Model")

st.write(
    """
The spatial super-resolution experiment uses an
SRGAN-style architecture adapted for numerical AQI
surfaces.

The experiment performs **4× spatial upscaling**:

**8 × 8 coarse AQI surface → 32 × 32 enhanced AQI surface**

The generator begins with bicubic upscaling and learns
residual spatial corrections. A discriminator is then
used during adversarial training to encourage generated
spatial patterns to resemble the target surfaces.
"""
)


col1, col2, col3 = st.columns(3)

col1.metric(
    "Bicubic MAE",
    "0.872 AQI"
)

col2.metric(
    "Generator V2 MAE",
    "0.263 AQI"
)

col3.metric(
    "SRGAN MAE",
    "0.256 AQI"
)


st.success(
    "On the experimental test dataset, the SRGAN "
    "reduced MAE by approximately 70.66% compared "
    "with bicubic interpolation."
)

st.caption(
    "SRGAN also achieved approximately 2.78% lower "
    "MAE than the non-adversarial Generator V2."
)


st.warning(
    "These results are measured on interpolated modelled "
    "AQI surfaces. They are not validation against true "
    "street-level high-resolution sensor ground truth."
)


# ==================================================
# DATA SOURCES
# ==================================================

st.divider()

st.subheader("📡 Data Sources")

st.markdown(
    """
**OpenAQ**

Used to search for available physical air-quality
ground stations.

**Open-Meteo Air Quality API / CAMS**

Used as a fallback when suitable fresh ground-station
measurements are unavailable.

**Open-Meteo Weather API**

Provides temperature, humidity, precipitation and
wind information required by the forecasting model.

**Historical Multi-City Dataset**

Historical AQI values from 10 representative Indian
cities are also used for LSTM forecasting experiments
and for constructing the interpolated spatial surfaces
used by the SRGAN experiment.
"""
)


# ==================================================
# AQI NOTE
# ==================================================

st.divider()

st.subheader("⚠️ Important AQI Note")

st.warning(
    "AQI produced from Open-Meteo/CAMS pollution data "
    "is a CPCB-style AQI calculated by this prototype. "
    "It must not be presented as an official CPCB "
    "station AQI."
)


# ==================================================
# CURRENT PROTOTYPE VS FUTURE WORK
# ==================================================

st.divider()

st.subheader("🚀 Current Prototype & Future Scope")

col1, col2 = st.columns(2)


with col1:

    st.write(
        "### ✅ Currently Implemented"
    )

    st.markdown(
        """
- Any-city search
- Current AQI monitoring
- OpenAQ ground-station checking
- CAMS fallback
- Multi-City LSTM forecasting
- 24h and 48h prediction
- Hotspot detection
- Red-zone classification
- Plantation priority
- Smart recommendations
- Citizen reporting
- Interactive maps
- Model-performance analysis
- 8 × 8 to 32 × 32 AQI spatial enhancement
- Residual super-resolution generator
- Adversarial discriminator training
- SRGAN model evaluation
- Spatial super-resolution comparison visualization
"""
    )


with col2:

    st.write(
        "### 🔮 Future Improvements"
    )

    st.markdown(
        """
- Validation using true high-resolution sensor ground-truth maps
- Higher-density ground monitoring stations
- Satellite-image and aerosol-data integration
- More training cities
- Improved forecasting accuracy
- Street-level sensor integration
- Traffic-data integration
- City-specific high-resolution SRGAN training
- Cloud deployment
- Automated pollution alerts
"""
    )


# ==================================================
# LIMITATIONS
# ==================================================

st.divider()

st.subheader("⚠️ Current Limitations")

st.markdown(
    """
- Ground-station data may be stale or incomplete.
- CAMS pollution data has coarse spatial resolution.
- Nearby locations can therefore show similar AQI values.
- The LSTM was trained using only 10 representative cities.
- Predictions for unseen cities are experimental.
- Forecast accuracy varies significantly between cities.
- The LSTM does not strongly outperform the persistence baseline overall.
- API availability and internet connectivity affect live operation.
- Citizen reports are unverified observations and do not modify AQI.
- The SRGAN experiment uses interpolated AQI surfaces created from 10 city-level locations.
- SRGAN targets are not true street-level high-resolution sensor measurements.
- SRGAN accuracy therefore represents experimental reconstruction performance, not validated street-level AQI accuracy.
"""
)


# ==================================================
# FINAL SUMMARY
# ==================================================

st.divider()

st.success(
    "The system combines air-quality monitoring, "
    "AI forecasting, experimental SRGAN-based spatial "
    "super-resolution, hotspot analysis, recommendation "
    "support and citizen participation into a single "
    "environmental intelligence prototype."
)