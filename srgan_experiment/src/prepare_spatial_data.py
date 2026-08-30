from pathlib import Path
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------
# Paths
# ---------------------------------------------------

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_city_final_aqi_dataset.csv"
)

OUTPUT_DIR = EXPERIMENT_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------
# Settings
# ---------------------------------------------------

CITY_ORDER = [
    "Belagavi",
    "Hubballi",
    "Bengaluru",
    "Delhi",
    "Mumbai",
    "Pune",
    "Hyderabad",
    "Chennai",
    "Kolkata",
    "Ahmedabad",
]

HR_SIZE = 32
LR_SIZE = 8
UPSCALE_FACTOR = HR_SIZE // LR_SIZE

IDW_POWER = 2.0
EPSILON = 1e-8


def build_idw_weights(city_coordinates):
    """
    Create fixed inverse-distance weights between every
    32x32 grid point and the 10 training cities.
    """

    latitudes = city_coordinates[:, 0]
    longitudes = city_coordinates[:, 1]

    lat_min = latitudes.min()
    lat_max = latitudes.max()
    lon_min = longitudes.min()
    lon_max = longitudes.max()

    grid_latitudes = np.linspace(lat_min, lat_max, HR_SIZE)
    grid_longitudes = np.linspace(lon_min, lon_max, HR_SIZE)

    grid_lon, grid_lat = np.meshgrid(
        grid_longitudes,
        grid_latitudes
    )

    grid_points = np.column_stack(
        [
            grid_lat.ravel(),
            grid_lon.ravel(),
        ]
    )

    # Approximate longitude correction for geographic distance
    mean_latitude = np.mean(latitudes)
    longitude_scale = np.cos(np.radians(mean_latitude))

    grid_lat = grid_points[:, 0][:, None]
    grid_lon = grid_points[:, 1][:, None]

    city_lat = latitudes[None, :]
    city_lon = longitudes[None, :]

    lat_difference = grid_lat - city_lat

    lon_difference = (
        grid_lon - city_lon
    ) * longitude_scale

    distance = np.sqrt(
        lat_difference ** 2
        + lon_difference ** 2
    )

    weights = 1.0 / (
        np.power(distance, IDW_POWER)
        + EPSILON
    )

    weights = weights / weights.sum(
        axis=1,
        keepdims=True
    )

    metadata = {
        "hr_size": HR_SIZE,
        "lr_size": LR_SIZE,
        "upscale_factor": UPSCALE_FACTOR,
        "latitude_min": float(lat_min),
        "latitude_max": float(lat_max),
        "longitude_min": float(lon_min),
        "longitude_max": float(lon_max),
        "idw_power": IDW_POWER,
        "city_order": CITY_ORDER,
    }

    return weights.astype(np.float32), metadata


def downsample_to_low_resolution(hr_maps):
    """
    Average-pool 32x32 maps into 8x8 maps.
    """

    block_size = UPSCALE_FACTOR

    number_of_maps = hr_maps.shape[0]

    lr_maps = hr_maps.reshape(
        number_of_maps,
        LR_SIZE,
        block_size,
        LR_SIZE,
        block_size,
    ).mean(axis=(2, 4))

    return lr_maps.astype(np.float32)


def normalize_aqi(data):
    """
    AQI range 0-500 -> [-1, 1]
    """

    data = np.clip(data, 0.0, 500.0)

    return (
        data / 250.0
    ) - 1.0


def main():

    print("\nSRGAN SPATIAL DATA PREPARATION\n")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "city",
        "latitude",
        "longitude",
        "time",
        "AQI",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "time",
            "city",
            "latitude",
            "longitude",
            "AQI",
        ]
    )

    # ---------------------------------------------------
    # City coordinates
    # ---------------------------------------------------

    city_coordinate_df = (
        df[
            [
                "city",
                "latitude",
                "longitude",
            ]
        ]
        .drop_duplicates(subset=["city"])
        .set_index("city")
        .reindex(CITY_ORDER)
    )

    if city_coordinate_df.isnull().any().any():
        raise ValueError(
            "One or more required training cities are missing."
        )

    city_coordinates = (
        city_coordinate_df[
            ["latitude", "longitude"]
        ]
        .to_numpy(dtype=np.float32)
    )

    print(
        f"Spatial locations: "
        f"{len(city_coordinates)}"
    )

    # ---------------------------------------------------
    # Time synchronized city AQI matrix
    # ---------------------------------------------------

    print(
        "Creating synchronized AQI time matrix..."
    )

    aqi_pivot = df.pivot_table(
        index="time",
        columns="city",
        values="AQI",
        aggfunc="mean",
    )

    aqi_pivot = (
        aqi_pivot
        .reindex(columns=CITY_ORDER)
        .dropna()
        .sort_index()
    )

    print(
        f"Usable timestamps: "
        f"{len(aqi_pivot)}"
    )

    aqi_matrix = aqi_pivot.to_numpy(
        dtype=np.float32
    )

    # ---------------------------------------------------
    # Create interpolated 32x32 AQI surfaces
    # ---------------------------------------------------

    print(
        "Creating 32x32 interpolated AQI maps..."
    )

    weights, metadata = build_idw_weights(
        city_coordinates
    )

    hr_flat = (
        aqi_matrix
        @ weights.T
    )

    hr_maps = hr_flat.reshape(
        -1,
        HR_SIZE,
        HR_SIZE,
    ).astype(np.float32)

    # ---------------------------------------------------
    # Generate 8x8 low-resolution maps
    # ---------------------------------------------------

    print(
        "Creating 8x8 low-resolution maps..."
    )

    lr_maps = downsample_to_low_resolution(
        hr_maps
    )

    # ---------------------------------------------------
    # Normalize for neural-network training
    # ---------------------------------------------------

    hr_normalized = normalize_aqi(
        hr_maps
    )[..., np.newaxis]

    lr_normalized = normalize_aqi(
        lr_maps
    )[..., np.newaxis]

    # ---------------------------------------------------
    # Chronological split
    # 70% train, 15% validation, 15% test
    # ---------------------------------------------------

    total_samples = len(hr_maps)

    train_end = int(
        total_samples * 0.70
    )

    validation_end = int(
        total_samples * 0.85
    )

    X_train = lr_normalized[:train_end]
    y_train = hr_normalized[:train_end]

    X_validation = lr_normalized[
        train_end:validation_end
    ]

    y_validation = hr_normalized[
        train_end:validation_end
    ]

    X_test = lr_normalized[
        validation_end:
    ]

    y_test = hr_normalized[
        validation_end:
    ]

    # ---------------------------------------------------
    # Save arrays
    # ---------------------------------------------------

    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_train,
    )

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y_train,
    )

    np.save(
        OUTPUT_DIR / "X_validation.npy",
        X_validation,
    )

    np.save(
        OUTPUT_DIR / "y_validation.npy",
        y_validation,
    )

    np.save(
        OUTPUT_DIR / "X_test.npy",
        X_test,
    )

    np.save(
        OUTPUT_DIR / "y_test.npy",
        y_test,
    )

    # Save original AQI test maps for later visualization
    np.save(
        OUTPUT_DIR / "hr_test_original.npy",
        hr_maps[validation_end:],
    )

    np.save(
        OUTPUT_DIR / "lr_test_original.npy",
        lr_maps[validation_end:],
    )

    # ---------------------------------------------------
    # Save timestamps
    # ---------------------------------------------------

    timestamp_df = pd.DataFrame(
        {
            "time": aqi_pivot.index
        }
    )

    timestamp_df.to_csv(
        OUTPUT_DIR / "timestamps.csv",
        index=False,
    )

    # ---------------------------------------------------
    # Save metadata
    # ---------------------------------------------------

    metadata.update(
        {
            "total_samples": int(
                total_samples
            ),
            "train_samples": int(
                len(X_train)
            ),
            "validation_samples": int(
                len(X_validation)
            ),
            "test_samples": int(
                len(X_test)
            ),
            "data_description":
                "IDW-interpolated AQI spatial surfaces "
                "created from 10 city-level modelled AQI "
                "time series for SRGAN prototype training.",
        }
    )

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\nDATA PREPARATION COMPLETE")
    print("-------------------------")

    print(
        "X_train:",
        X_train.shape,
    )

    print(
        "y_train:",
        y_train.shape,
    )

    print(
        "X_validation:",
        X_validation.shape,
    )

    print(
        "y_validation:",
        y_validation.shape,
    )

    print(
        "X_test:",
        X_test.shape,
    )

    print(
        "y_test:",
        y_test.shape,
    )

    print(
        "\nFiles saved to:",
        OUTPUT_DIR,
    )

    print(
        "\nIMPORTANT:"
        "\nThese 32x32 targets are interpolated "
        "modelled AQI surfaces, not true "
        "street-level sensor measurements."
    )


if __name__ == "__main__":
    main()