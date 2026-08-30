from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from srgan_models import PixelShuffle


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = EXPERIMENT_DIR / "data"
MODEL_DIR = EXPERIMENT_DIR / "models"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "generator_pretrained.keras"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def denormalize_aqi(data):
    """
    Convert normalized [-1, 1] values back to AQI 0-500.
    """

    data = (data + 1.0) * 250.0
    return np.clip(data, 0.0, 500.0)


def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))

    mse = np.mean(
        np.square(actual - predicted)
    )

    rmse = np.sqrt(mse)

    return mae, mse, rmse


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("\nPRETRAINED GENERATOR EVALUATION\n")

    # -----------------------------------------------------
    # Load test data
    # -----------------------------------------------------

    print("Loading test data...")

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    )

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # -----------------------------------------------------
    # Load pretrained generator
    # -----------------------------------------------------

    print("\nLoading pretrained generator...")

    generator = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "PixelShuffle": PixelShuffle
        },
        compile=False
    )

    # -----------------------------------------------------
    # Generator prediction
    # -----------------------------------------------------

    print("Generating SR predictions...")

    sr_predictions = generator.predict(
        X_test,
        batch_size=32,
        verbose=1
    )

    # -----------------------------------------------------
    # Bicubic baseline
    # -----------------------------------------------------

    print("\nCreating bicubic baseline...")

    bicubic_predictions = tf.image.resize(
        X_test,
        size=(32, 32),
        method="bicubic"
    ).numpy()

    # -----------------------------------------------------
    # Convert back to original AQI scale
    # -----------------------------------------------------

    actual_aqi = denormalize_aqi(
        y_test
    )

    sr_aqi = denormalize_aqi(
        sr_predictions
    )

    bicubic_aqi = denormalize_aqi(
        bicubic_predictions
    )

    # -----------------------------------------------------
    # Calculate metrics
    # -----------------------------------------------------

    sr_mae, sr_mse, sr_rmse = (
        calculate_metrics(
            actual_aqi,
            sr_aqi
        )
    )

    bic_mae, bic_mse, bic_rmse = (
        calculate_metrics(
            actual_aqi,
            bicubic_aqi
        )
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print("\nEVALUATION RESULTS")
    print("------------------")

    print("\nPretrained Generator:")
    print(f"MAE  : {sr_mae:.3f} AQI")
    print(f"MSE  : {sr_mse:.3f}")
    print(f"RMSE : {sr_rmse:.3f} AQI")

    print("\nBicubic Baseline:")
    print(f"MAE  : {bic_mae:.3f} AQI")
    print(f"MSE  : {bic_mse:.3f}")
    print(f"RMSE : {bic_rmse:.3f} AQI")

    # -----------------------------------------------------
    # Comparison
    # -----------------------------------------------------

    if sr_mae < bic_mae:

        improvement = (
            (bic_mae - sr_mae)
            / bic_mae
        ) * 100

        print(
            f"\nGenerator MAE improvement "
            f"over bicubic: {improvement:.2f}%"
        )

    else:

        difference = (
            (sr_mae - bic_mae)
            / bic_mae
        ) * 100

        print(
            f"\nBicubic currently performs "
            f"{difference:.2f}% better in MAE."
        )

    # -----------------------------------------------------
    # Save result CSV
    # -----------------------------------------------------

    results = pd.DataFrame(
        {
            "Method": [
                "Pretrained Generator",
                "Bicubic Baseline"
            ],
            "MAE_AQI": [
                sr_mae,
                bic_mae
            ],
            "MSE": [
                sr_mse,
                bic_mse
            ],
            "RMSE_AQI": [
                sr_rmse,
                bic_rmse
            ]
        }
    )

    results.to_csv(
        OUTPUT_DIR
        / "pretrained_evaluation.csv",
        index=False
    )

    # -----------------------------------------------------
    # Save predictions for later visualization
    # -----------------------------------------------------

    np.save(
        OUTPUT_DIR / "sr_predictions.npy",
        sr_aqi
    )

    np.save(
        OUTPUT_DIR / "bicubic_predictions.npy",
        bicubic_aqi
    )

    np.save(
        OUTPUT_DIR / "actual_test_maps.npy",
        actual_aqi
    )

    print(
        "\nResults saved to:",
        OUTPUT_DIR
        / "pretrained_evaluation.csv"
    )

    print(
        "\nEvaluation complete."
    )


if __name__ == "__main__":
    main()