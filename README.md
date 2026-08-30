# AI-Powered Air Quality Intelligence System

An AI-based air-quality intelligence prototype that provides current AQI monitoring, 24-hour and 48-hour forecasting, pollution hotspot detection, red-zone classification, plantation priorities, citizen reporting, intelligent recommendations, and experimental SRGAN-based spatial super-resolution.

## Project Overview

Air pollution monitoring systems often depend on limited ground monitoring stations. Atmospheric pollution datasets can also have coarse spatial resolution.

This project combines available ground-station measurements, modelled atmospheric air-quality data, weather information, machine-learning forecasting, hotspot analysis, and experimental spatial super-resolution in a single Streamlit-based decision-support system.

Users can search for a city and view current air-quality conditions, AI forecasts, hotspots, recommendations, maps, model performance, and spatial super-resolution results.

## Main Features

- Search any city
- Current AQI calculation
- Current pollutant levels
- 24-hour AQI prediction
- 48-hour AQI prediction
- Multi-City LSTM forecasting
- Dynamic pollution hotspot detection
- Red-zone classification
- Plantation priority recommendations
- Pollutant-specific recommendations
- Ground-station availability checking
- Open-Meteo / CAMS fallback
- Citizen pollution reporting
- Interactive pollution hotspot map
- Model performance analysis
- 8 × 8 to 32 × 32 spatial AQI enhancement
- Residual super-resolution generator
- Adversarial discriminator training
- SRGAN model evaluation
- Spatial super-resolution comparison visualization

## Current Air Quality

The system calculates CPCB-style AQI using pollutant concentrations.

Supported pollutants include:

- PM2.5
- PM10
- NO2
- SO2
- Ozone
- CO

The dashboard displays:

- AQI
- AQI Category
- Dominant Pollutant
- Pollution Zone
- Pollutant Concentrations
- Latest Data Timestamp
- Data Source

## Dynamic Data Source Selection

The system first checks for nearby OpenAQ ground monitoring stations.

If sufficiently fresh and complete ground-station measurements are available, they can be used.

If the ground-station data is stale, incomplete, unavailable, or does not contain sufficient hourly coverage, the system automatically falls back to Open-Meteo / CAMS atmospheric data.

## AI AQI Forecasting

A Multi-City LSTM model is used to predict AQI for:

- 24 hours ahead
- 48 hours ahead

The model uses historical air-quality and weather information such as:

- Historical AQI
- PM2.5
- PM10
- CO
- NO2
- SO2
- Ozone
- Temperature
- Relative Humidity
- Precipitation
- Wind Speed
- Time-based features
- AQI trend information
- Future weather information
- Latitude
- Longitude

## Training Cities

The forecasting model was trained using historical data from 10 representative Indian cities:

1. Belagavi
2. Hubballi
3. Bengaluru
4. Delhi
5. Mumbai
6. Pune
7. Hyderabad
8. Chennai
9. Kolkata
10. Ahmedabad

The system can also generate predictions for other geocodable cities.

Forecasts for cities outside the training set should be considered generalized experimental predictions.

## LSTM Model Performance

The Multi-City LSTM was compared with a persistence baseline.

| Forecast | LSTM MAE | Persistence Baseline MAE |
|---|---:|---:|
| 24 Hour | 12.39 | 12.30 |
| 48 Hour | 16.66 | 16.70 |

The LSTM and persistence baseline perform similarly overall.

The project does not claim that the LSTM strongly outperforms the persistence baseline in every city.

## Spatial Super-Resolution

The project includes an experimental SRGAN-style AQI spatial super-resolution module.

The module performs:

**8 × 8 coarse AQI surface → 32 × 32 enhanced AQI surface**

The spatial experiment uses:

- Inverse-distance interpolation
- Bicubic upscaling baseline
- Residual super-resolution generator
- Discriminator
- Content loss
- Adversarial loss

The generator first learns corrections over a bicubic-upscaled surface. Adversarial training is then used to encourage generated spatial patterns to resemble the target AQI surfaces.

## SRGAN Evaluation

Final test-set results:

| Method | MAE (AQI) | RMSE (AQI) |
|---|---:|---:|
| Bicubic Baseline | 0.872 | 2.047 |
| Generator V2 | 0.263 | 0.779 |
| SRGAN Generator | 0.256 | 0.712 |

The SRGAN Generator achieved approximately:

- **70.66% lower MAE than bicubic interpolation**
- **2.78% lower MAE than Generator V2**

The SRGAN Generator was the best method on the experimental test dataset.

### Important SRGAN Limitation

The SRGAN target surfaces are not true street-level high-resolution sensor maps.

The original historical dataset contains one fixed geographic coordinate for each of 10 cities. Synchronized AQI values were converted into 32 × 32 spatial surfaces using inverse-distance interpolation.

These surfaces were then downsampled to 8 × 8 and used for super-resolution training.

Therefore, SRGAN performance represents reconstruction accuracy on interpolated modelled AQI surfaces and should not be interpreted as validated street-level AQI accuracy.

## Dynamic Pollution Hotspot Detection

The system scans nine points around the selected city:

- Center
- North
- South
- East
- West
- North-East
- North-West
- South-East
- South-West

Each point is classified using project-defined thresholds:

| AQI Range | Zone |
|---|---|
| 0 - 100 | Normal |
| 101 - 200 | Pollution Hotspot |
| 201 - 300 | Red Zone |
| Above 300 | Critical Red Zone |

These hotspot classifications are project-defined decision-support thresholds and are not official CPCB red-zone definitions.

## Plantation and Pollution Recommendations

The system generates rule-based recommendations using:

- AQI
- Pollution zone
- Dominant pollutant
- Hotspot condition

Plantation priorities include:

- Low
- Moderate
- High
- Urgent

Recommendations may include:

- Roadside plantation
- Green buffers
- Traffic management
- Road-dust control
- Construction-dust control
- Waste-burning prevention
- Industrial emission control

Green buffers are treated as supporting long-term measures and not replacements for direct emission control.

## Citizen Engagement

Users can submit pollution observations including:

- Heavy Traffic
- Road Dust
- Construction Dust
- Waste Burning
- Industrial Smoke
- Vehicle Smoke
- Other

Citizen reports are supplementary, unverified observations and do not directly modify calculated AQI values.

## Technologies Used

### Programming

- Python

### Machine Learning

- TensorFlow
- Keras
- Scikit-learn
- NumPy
- Pandas

### Visualization

- Matplotlib
- Plotly
- Folium

### Web Application

- Streamlit
- Streamlit-Folium

### APIs and Data Sources

- OpenAQ
- Open-Meteo
- CAMS atmospheric air-quality data
- Open-Meteo Weather API
- Geocoding APIs

## Project Structure

```text
Air Quality Intelligence/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── model_ready_multi_city/
│
├── models/
│   ├── multi_city_lstm.keras
│   ├── multi_feature_scaler.pkl
│   ├── multi_aux_scaler.pkl
│   └── multi_target_scaler.pkl
│
├── outputs/
│   ├── maps/
│   ├── plots/
│   └── multi_city_model_results.csv
│
├── pages/
│   ├── 2_Citizen_Engagement.py
│   ├── 3_Model_Performance.py
│   ├── 4_About_Project.py
│   └── 5_Spatial_Super_Resolution.py
│
├── srgan_experiment/
│   ├── data/
│   ├── models/
│   ├── outputs/
│   └── src/
│
└── src/
    ├── live_air_quality.py
    ├── data_source_manager.py
    ├── multi_city_live_prediction.py
    ├── dynamic_hotspot_detection.py
    ├── hotspot_recommendations.py
    ├── citizen_reports.py
    └── other preprocessing, training and evaluation scripts