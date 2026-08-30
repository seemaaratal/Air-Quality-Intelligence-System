from pathlib import Path

import numpy as np
import tensorflow as tf

from srgan_models import build_generator


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = EXPERIMENT_DIR / "data"
MODEL_DIR = EXPERIMENT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Training Settings
# ---------------------------------------------------------

EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4


def main():

    print("\nSRGAN GENERATOR PRE-TRAINING\n")

    # -----------------------------------------------------
    # Load prepared data
    # -----------------------------------------------------

    print("Loading training data...")

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    )

    y_train = np.load(
        DATA_DIR / "y_train.npy"
    )

    X_validation = np.load(
        DATA_DIR / "X_validation.npy"
    )

    y_validation = np.load(
        DATA_DIR / "y_validation.npy"
    )

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print(
        "X_validation:",
        X_validation.shape
    )

    print(
        "y_validation:",
        y_validation.shape
    )

    # -----------------------------------------------------
    # Build generator
    # -----------------------------------------------------

    print("\nBuilding generator...")

    generator = build_generator()

    generator.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="mse",
        metrics=["mae"]
    )

    # -----------------------------------------------------
    # Callbacks
    # -----------------------------------------------------

    best_model_path = (
        MODEL_DIR
        / "generator_pretrained.keras"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    # -----------------------------------------------------
    # Pre-train
    # -----------------------------------------------------

    print("\nStarting generator pre-training...\n")

    history = generator.fit(
        X_train,
        y_train,
        validation_data=(
            X_validation,
            y_validation
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    # -----------------------------------------------------
    # Final validation check
    # -----------------------------------------------------

    validation_loss, validation_mae = (
        generator.evaluate(
            X_validation,
            y_validation,
            verbose=0
        )
    )

    print("\nPRE-TRAINING COMPLETE")
    print("---------------------")

    print(
        f"Validation MSE: "
        f"{validation_loss:.6f}"
    )

    print(
        f"Validation MAE: "
        f"{validation_mae:.6f}"
    )

    print(
        "\nBest generator saved to:"
    )

    print(best_model_path)

    print(
        "\nNOTE:"
        "\nThis generator is learning reconstruction "
        "of interpolated AQI surfaces."
    )


if __name__ == "__main__":
    main()