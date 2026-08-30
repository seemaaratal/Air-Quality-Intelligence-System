from pathlib import Path

import numpy as np
import tensorflow as tf

from srgan_models_v2 import build_generator_v2


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = EXPERIMENT_DIR / "data"
MODEL_DIR = EXPERIMENT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = (
    MODEL_DIR / "generator_v2_best.keras"
)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 5e-5


# ---------------------------------------------------------
# Custom callback
# Keeps bicubic-level initial model safe
# ---------------------------------------------------------

class BestValidationMAESaver(tf.keras.callbacks.Callback):

    def __init__(self, filepath, initial_best):
        super().__init__()

        self.filepath = filepath
        self.best = initial_best

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        current = logs.get("val_mae")

        if current is None:
            return

        if current < self.best:

            print(
                f"\nValidation MAE improved "
                f"from {self.best:.6f} "
                f"to {current:.6f}"
            )

            self.best = current

            self.model.save(
                self.filepath
            )

            print(
                "Best V2 generator saved."
            )


def main():

    print("\nGENERATOR V2 TRAINING\n")

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    print("Loading data...")

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

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "X_validation:",
        X_validation.shape
    )

    print(
        "y_validation:",
        y_validation.shape
    )

    # -----------------------------------------------------
    # Build model
    # -----------------------------------------------------

    print("\nBuilding Generator V2...")

    generator = build_generator_v2()

    generator.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="mae",
        metrics=["mae"]
    )

    # -----------------------------------------------------
    # Initial bicubic-level validation score
    # -----------------------------------------------------

    initial_loss, initial_mae = (
        generator.evaluate(
            X_validation,
            y_validation,
            batch_size=BATCH_SIZE,
            verbose=0
        )
    )

    print(
        "\nInitial validation MAE "
        f"(bicubic starting point): "
        f"{initial_mae:.6f}"
    )

    # Save starting model first
    generator.save(
        BEST_MODEL_PATH
    )

    print(
        "Initial V2 model saved safely."
    )

    # -----------------------------------------------------
    # Callbacks
    # -----------------------------------------------------

    callbacks = [

        BestValidationMAESaver(
            filepath=BEST_MODEL_PATH,
            initial_best=initial_mae
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae",
            patience=6,
            mode="min",
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            mode="min",
            verbose=1
        ),
    ]

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    print(
        "\nStarting V2 residual learning...\n"
    )

    generator.fit(
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
    # Load best saved model
    # -----------------------------------------------------

    print(
        "\nLoading best V2 model..."
    )

    best_generator = (
        tf.keras.models.load_model(
            BEST_MODEL_PATH,
            compile=False
        )
    )

    best_generator.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="mae",
        metrics=["mae"]
    )

    best_loss, best_mae = (
        best_generator.evaluate(
            X_validation,
            y_validation,
            batch_size=BATCH_SIZE,
            verbose=0
        )
    )

    print(
        "\nV2 TRAINING COMPLETE"
    )

    print(
        "--------------------"
    )

    print(
        f"Initial Validation MAE: "
        f"{initial_mae:.6f}"
    )

    print(
        f"Best Validation MAE: "
        f"{best_mae:.6f}"
    )

    if best_mae < initial_mae:

        improvement = (
            (initial_mae - best_mae)
            / initial_mae
        ) * 100

        print(
            f"Validation improvement: "
            f"{improvement:.2f}%"
        )

    else:

        print(
            "No improvement over the "
            "bicubic starting point."
        )

    print(
        "\nBest model:"
    )

    print(
        BEST_MODEL_PATH
    )


if __name__ == "__main__":
    main()