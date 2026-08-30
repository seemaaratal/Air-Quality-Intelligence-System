from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Local imports
# ---------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from srgan_models_v2 import BicubicResize, ClipAQI


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = EXPERIMENT_DIR / "data"
MODEL_DIR = EXPERIMENT_DIR / "models"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = (
    MODEL_DIR / "srgan_generator_v2_best.keras"
)

OUTPUT_IMAGE = (
    OUTPUT_DIR / "srgan_comparison.png"
)


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def denormalize_aqi(data):

    return np.clip(
        (data + 1.0) * 250.0,
        0.0,
        500.0
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("\nSRGAN RESULT VISUALIZATION\n")
    print("--------------------------")

    # -----------------------------------------------------
    # Load test data
    # -----------------------------------------------------

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    ).astype(np.float32)

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    ).astype(np.float32)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # -----------------------------------------------------
    # Select an informative test sample
    # -----------------------------------------------------

    actual_all = denormalize_aqi(
        y_test
    )

    flattened = actual_all.reshape(
        len(actual_all),
        -1
    )

    spatial_ranges = np.ptp(
        flattened,
        axis=1
    )

    sample_index = int(
        np.argmax(spatial_ranges)
    )

    print(
        "\nSelected test sample:",
        sample_index
    )

    # -----------------------------------------------------
    # Load trained SRGAN generator
    # -----------------------------------------------------

    print(
        "Loading best SRGAN generator..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "BicubicResize": BicubicResize,
            "ClipAQI": ClipAQI
        },
        compile=False
    )

    print(
        "Model loaded successfully."
    )

    # -----------------------------------------------------
    # Prepare selected input
    # -----------------------------------------------------

    low_resolution = X_test[
        sample_index:sample_index + 1
    ]

    target = y_test[
        sample_index:sample_index + 1
    ]

    # -----------------------------------------------------
    # SRGAN prediction
    # -----------------------------------------------------

    print(
        "Generating SRGAN output..."
    )

    srgan_prediction = model.predict(
        low_resolution,
        verbose=0
    )

    # -----------------------------------------------------
    # Bicubic baseline
    # -----------------------------------------------------

    bicubic_prediction = tf.image.resize(
        low_resolution,
        size=(32, 32),
        method="bicubic"
    ).numpy()

    bicubic_prediction = np.clip(
        bicubic_prediction,
        -1.0,
        1.0
    )

    # -----------------------------------------------------
    # Convert back to AQI
    # -----------------------------------------------------

    low_aqi = denormalize_aqi(
        low_resolution
    )[0, :, :, 0]

    bicubic_aqi = denormalize_aqi(
        bicubic_prediction
    )[0, :, :, 0]

    srgan_aqi = denormalize_aqi(
        srgan_prediction
    )[0, :, :, 0]

    target_aqi = denormalize_aqi(
        target
    )[0, :, :, 0]

    # -----------------------------------------------------
    # Sample-level MAE
    # -----------------------------------------------------

    bicubic_mae = np.mean(
        np.abs(
            target_aqi
            - bicubic_aqi
        )
    )

    srgan_mae = np.mean(
        np.abs(
            target_aqi
            - srgan_aqi
        )
    )

    print(
        f"\nSample Bicubic MAE: "
        f"{bicubic_mae:.3f} AQI"
    )

    print(
        f"Sample SRGAN MAE: "
        f"{srgan_mae:.3f} AQI"
    )

    # -----------------------------------------------------
    # Timestamp
    # -----------------------------------------------------

    timestamp_text = ""

    timestamp_file = (
        DATA_DIR / "timestamps.csv"
    )

    if timestamp_file.exists():

        timestamps = pd.read_csv(
            timestamp_file
        )

        test_start = (
            len(timestamps)
            - len(X_test)
        )

        global_index = (
            test_start
            + sample_index
        )

        if global_index < len(timestamps):

            timestamp_text = str(
                timestamps.iloc[
                    global_index
                ]["time"]
            )

            print(
                "Timestamp:",
                timestamp_text
            )

    # -----------------------------------------------------
    # Common AQI scale
    # -----------------------------------------------------

    vmin = min(
        np.min(low_aqi),
        np.min(bicubic_aqi),
        np.min(srgan_aqi),
        np.min(target_aqi)
    )

    vmax = max(
        np.max(low_aqi),
        np.max(bicubic_aqi),
        np.max(srgan_aqi),
        np.max(target_aqi)
    )

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5),
        constrained_layout=True
    )

    # Low Resolution
    im0 = axes[0].imshow(
        low_aqi,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )

    axes[0].set_title(
        "Low Resolution\n8 × 8"
    )

    # Bicubic
    axes[1].imshow(
        bicubic_aqi,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax
    )

    axes[1].set_title(
        f"Bicubic\n32 × 32\n"
        f"MAE = {bicubic_mae:.3f}"
    )

    # SRGAN
    axes[2].imshow(
        srgan_aqi,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax
    )

    axes[2].set_title(
        f"SRGAN\n32 × 32\n"
        f"MAE = {srgan_mae:.3f}"
    )

    # Target
    axes[3].imshow(
        target_aqi,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax
    )

    axes[3].set_title(
        "Target AQI Surface\n32 × 32"
    )

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    colorbar = fig.colorbar(
        im0,
        ax=axes,
        shrink=0.85
    )

    colorbar.set_label(
        "AQI"
    )

    title = (
        "AQI Spatial Super-Resolution Comparison"
    )

    if timestamp_text:
        title += (
            f"\nTimestamp: {timestamp_text}"
        )

    fig.suptitle(
        title,
        fontsize=15
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    plt.savefig(
        OUTPUT_IMAGE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nComparison image saved to:"
    )

    print(
        OUTPUT_IMAGE
    )

    print(
        "\nVisualization complete."
    )

    print(
        "\nNOTE:"
        "\nThe target is an interpolated "
        "modelled AQI surface, not a true "
        "street-level sensor ground-truth map."
    )


if __name__ == "__main__":
    main()