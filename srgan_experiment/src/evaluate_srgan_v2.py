from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tensorflow as tf


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

V2_MODEL_PATH = (
    MODEL_DIR / "generator_v2_best.keras"
)

SRGAN_MODEL_PATH = (
    MODEL_DIR / "srgan_generator_v2_best.keras"
)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def denormalize_aqi(data):
    """
    Convert normalized [-1, 1] values to AQI 0-500.
    """

    return np.clip(
        (data + 1.0) * 250.0,
        0.0,
        500.0
    )


def calculate_metrics(actual, predicted):

    mae = np.mean(
        np.abs(actual - predicted)
    )

    mse = np.mean(
        np.square(actual - predicted)
    )

    rmse = np.sqrt(mse)

    return mae, mse, rmse


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("\nFINAL SRGAN TEST EVALUATION\n")
    print("---------------------------")

    # -----------------------------------------------------
    # Load Test Data
    # -----------------------------------------------------

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    ).astype(np.float32)

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    ).astype(np.float32)

    print(
        "X_test:",
        X_test.shape
    )

    print(
        "y_test:",
        y_test.shape
    )

    # -----------------------------------------------------
    # Load Generator V2
    # -----------------------------------------------------

    print(
        "\nLoading Generator V2..."
    )

    generator_v2 = tf.keras.models.load_model(
        V2_MODEL_PATH,
        custom_objects={
            "BicubicResize": BicubicResize,
            "ClipAQI": ClipAQI
        },
        compile=False
    )

    print(
        "Generator V2 loaded."
    )

    # -----------------------------------------------------
    # Load SRGAN Generator
    # -----------------------------------------------------

    print(
        "\nLoading SRGAN Generator..."
    )

    srgan_generator = (
        tf.keras.models.load_model(
            SRGAN_MODEL_PATH,
            custom_objects={
                "BicubicResize": BicubicResize,
                "ClipAQI": ClipAQI
            },
            compile=False
        )
    )

    print(
        "SRGAN Generator loaded."
    )

    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    print(
        "\nGenerating Generator V2 predictions..."
    )

    v2_predictions = (
        generator_v2.predict(
            X_test,
            batch_size=32,
            verbose=1
        )
    )

    print(
        "\nGenerating SRGAN predictions..."
    )

    srgan_predictions = (
        srgan_generator.predict(
            X_test,
            batch_size=32,
            verbose=1
        )
    )

    # -----------------------------------------------------
    # Bicubic Baseline
    # -----------------------------------------------------

    print(
        "\nCreating bicubic baseline..."
    )

    bicubic_predictions = (
        tf.image.resize(
            X_test,
            size=(32, 32),
            method="bicubic"
        ).numpy()
    )

    bicubic_predictions = np.clip(
        bicubic_predictions,
        -1.0,
        1.0
    )

    # -----------------------------------------------------
    # Convert to Original AQI Scale
    # -----------------------------------------------------

    actual_aqi = denormalize_aqi(
        y_test
    )

    bicubic_aqi = denormalize_aqi(
        bicubic_predictions
    )

    v2_aqi = denormalize_aqi(
        v2_predictions
    )

    srgan_aqi = denormalize_aqi(
        srgan_predictions
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    bic_mae, bic_mse, bic_rmse = (
        calculate_metrics(
            actual_aqi,
            bicubic_aqi
        )
    )

    v2_mae, v2_mse, v2_rmse = (
        calculate_metrics(
            actual_aqi,
            v2_aqi
        )
    )

    srgan_mae, srgan_mse, srgan_rmse = (
        calculate_metrics(
            actual_aqi,
            srgan_aqi
        )
    )

    # -----------------------------------------------------
    # Display Results
    # -----------------------------------------------------

    print("\nFINAL TEST RESULTS")
    print("==================")

    print("\nBicubic Baseline")
    print("----------------")
    print(
        f"MAE  : {bic_mae:.3f} AQI"
    )
    print(
        f"MSE  : {bic_mse:.3f}"
    )
    print(
        f"RMSE : {bic_rmse:.3f} AQI"
    )

    print("\nGenerator V2")
    print("------------")
    print(
        f"MAE  : {v2_mae:.3f} AQI"
    )
    print(
        f"MSE  : {v2_mse:.3f}"
    )
    print(
        f"RMSE : {v2_rmse:.3f} AQI"
    )

    print("\nSRGAN Generator")
    print("---------------")
    print(
        f"MAE  : {srgan_mae:.3f} AQI"
    )
    print(
        f"MSE  : {srgan_mse:.3f}"
    )
    print(
        f"RMSE : {srgan_rmse:.3f} AQI"
    )

    # -----------------------------------------------------
    # Improvement Calculations
    # -----------------------------------------------------

    v2_vs_bicubic = (
        (bic_mae - v2_mae)
        / bic_mae
    ) * 100

    srgan_vs_bicubic = (
        (bic_mae - srgan_mae)
        / bic_mae
    ) * 100

    srgan_vs_v2 = (
        (v2_mae - srgan_mae)
        / v2_mae
    ) * 100

    print("\nIMPROVEMENT SUMMARY")
    print("-------------------")

    print(
        f"Generator V2 vs Bicubic: "
        f"{v2_vs_bicubic:.2f}%"
    )

    print(
        f"SRGAN vs Bicubic: "
        f"{srgan_vs_bicubic:.2f}%"
    )

    if srgan_mae < v2_mae:

        print(
            f"SRGAN vs Generator V2: "
            f"{srgan_vs_v2:.2f}% improvement"
        )

    else:

        difference = (
            (srgan_mae - v2_mae)
            / v2_mae
        ) * 100

        print(
            f"Generator V2 is "
            f"{difference:.2f}% better "
            f"than SRGAN in MAE."
        )

    # -----------------------------------------------------
    # Find Best Method
    # -----------------------------------------------------

    results = pd.DataFrame(
        {
            "Method": [
                "Bicubic Baseline",
                "Generator V2",
                "SRGAN Generator"
            ],
            "MAE_AQI": [
                bic_mae,
                v2_mae,
                srgan_mae
            ],
            "MSE": [
                bic_mse,
                v2_mse,
                srgan_mse
            ],
            "RMSE_AQI": [
                bic_rmse,
                v2_rmse,
                srgan_rmse
            ]
        }
    )

    best_method = (
        results
        .sort_values("MAE_AQI")
        .iloc[0]["Method"]
    )

    print(
        "\nBest test-set method:",
        best_method
    )

    # -----------------------------------------------------
    # Save Results
    # -----------------------------------------------------

    results_path = (
        OUTPUT_DIR
        / "final_srgan_evaluation.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    np.save(
        OUTPUT_DIR
        / "final_srgan_predictions.npy",
        srgan_aqi
    )

    print(
        "\nResults saved to:"
    )

    print(
        results_path
    )

    print(
        "\nIMPORTANT:"
        "\nEvaluation is performed on "
        "interpolated modelled AQI surfaces. "
        "It does not represent validation "
        "against true street-level "
        "high-resolution sensor maps."
    )


if __name__ == "__main__":
    main()