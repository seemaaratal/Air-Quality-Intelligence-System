from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

# Allow importing local SRGAN files
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from srgan_models_v2 import BicubicResize, ClipAQI


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = EXPERIMENT_DIR / "data"
MODEL_DIR = EXPERIMENT_DIR / "models"

MODEL_PATH = MODEL_DIR / "generator_v2_best.keras"


def denormalize(data):
    return np.clip(
        (data + 1.0) * 250.0,
        0.0,
        500.0
    )


def metrics(actual, predicted):

    mae = np.mean(
        np.abs(actual - predicted)
    )

    mse = np.mean(
        (actual - predicted) ** 2
    )

    rmse = np.sqrt(mse)

    return mae, mse, rmse


def main():

    print("\nGENERATOR V2 TEST EVALUATION\n")

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    )

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print("\nLoading best V2 generator...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "BicubicResize": BicubicResize,
            "ClipAQI": ClipAQI
        },
        compile=False
    )

    print("Model loaded successfully.")

    print("\nGenerating V2 predictions...")

    v2_prediction = model.predict(
        X_test,
        batch_size=32,
        verbose=1
    )

    print("\nCreating bicubic baseline...")

    bicubic_prediction = tf.image.resize(
        X_test,
        size=(32, 32),
        method="bicubic"
    ).numpy()

    bicubic_prediction = np.clip(
        bicubic_prediction,
        -1.0,
        1.0
    )

    # Convert to AQI scale
    actual_aqi = denormalize(y_test)
    v2_aqi = denormalize(v2_prediction)
    bicubic_aqi = denormalize(
        bicubic_prediction
    )

    # Metrics
    v2_mae, v2_mse, v2_rmse = metrics(
        actual_aqi,
        v2_aqi
    )

    bic_mae, bic_mse, bic_rmse = metrics(
        actual_aqi,
        bicubic_aqi
    )

    print("\nTEST RESULTS")
    print("------------")

    print("\nGenerator V2:")
    print(f"MAE  : {v2_mae:.3f} AQI")
    print(f"MSE  : {v2_mse:.3f}")
    print(f"RMSE : {v2_rmse:.3f} AQI")

    print("\nBicubic Baseline:")
    print(f"MAE  : {bic_mae:.3f} AQI")
    print(f"MSE  : {bic_mse:.3f}")
    print(f"RMSE : {bic_rmse:.3f} AQI")

    if v2_mae < bic_mae:

        improvement = (
            (bic_mae - v2_mae)
            / bic_mae
        ) * 100

        print(
            f"\nV2 MAE improvement over "
            f"bicubic: {improvement:.2f}%"
        )

    else:

        difference = (
            (v2_mae - bic_mae)
            / bic_mae
        ) * 100

        print(
            f"\nBicubic performs "
            f"{difference:.2f}% better."
        )


if __name__ == "__main__":
    main()