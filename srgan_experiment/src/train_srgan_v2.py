from pathlib import Path
import sys
import time

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

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRETRAINED_GENERATOR = (
    MODEL_DIR / "generator_v2_best.keras"
)

BEST_SR_MODEL = (
    MODEL_DIR / "srgan_generator_v2_best.keras"
)

FINAL_SR_MODEL = (
    MODEL_DIR / "srgan_generator_v2_final.keras"
)

DISCRIMINATOR_MODEL = (
    MODEL_DIR / "srgan_discriminator_v2.keras"
)


# ---------------------------------------------------------
# Training settings
# ---------------------------------------------------------

EPOCHS = 5
BATCH_SIZE = 16

GENERATOR_LR = 1e-5
DISCRIMINATOR_LR = 5e-5

# Small adversarial weight protects reconstruction accuracy
ADVERSARIAL_WEIGHT = 1e-4

SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------
# Discriminator
# 32x32 AQI map -> real/fake score
# ---------------------------------------------------------

def build_discriminator():

    inputs = tf.keras.Input(
        shape=(32, 32, 1),
        name="aqi_map"
    )

    x = tf.keras.layers.Conv2D(
        32,
        kernel_size=3,
        strides=1,
        padding="same"
    )(inputs)

    x = tf.keras.layers.LeakyReLU(
        negative_slope=0.2
    )(x)

    x = tf.keras.layers.Conv2D(
        32,
        kernel_size=3,
        strides=2,
        padding="same"
    )(x)

    x = tf.keras.layers.LeakyReLU(
        negative_slope=0.2
    )(x)

    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=3,
        strides=1,
        padding="same"
    )(x)

    x = tf.keras.layers.LeakyReLU(
        negative_slope=0.2
    )(x)

    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=3,
        strides=2,
        padding="same"
    )(x)

    x = tf.keras.layers.LeakyReLU(
        negative_slope=0.2
    )(x)

    x = tf.keras.layers.Conv2D(
        128,
        kernel_size=3,
        strides=1,
        padding="same"
    )(x)

    x = tf.keras.layers.LeakyReLU(
        negative_slope=0.2
    )(x)

    x = tf.keras.layers.Conv2D(
        128,
        kernel_size=3,
        strides=2,
        padding="same"
    )(x)

    x = tf.keras.layers.LeakyReLU(
        negative_slope=0.2
    )(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(
        128
    )(x)

    x = tf.keras.layers.LeakyReLU(
        negative_slope=0.2
    )(x)

    # Raw logit
    outputs = tf.keras.layers.Dense(
        1,
        name="real_fake_logit"
    )(x)

    return tf.keras.Model(
        inputs,
        outputs,
        name="AQI_SR_Discriminator_V2"
    )


# ---------------------------------------------------------
# Validation MAE
# ---------------------------------------------------------

def calculate_validation_mae(
    generator,
    validation_dataset
):

    total_error = 0.0
    total_values = 0

    for lr_images, hr_images in validation_dataset:

        predictions = generator(
            lr_images,
            training=False
        )

        absolute_error = tf.abs(
            hr_images - predictions
        )

        total_error += float(
            tf.reduce_sum(
                absolute_error
            ).numpy()
        )

        total_values += int(
            tf.size(
                absolute_error
            ).numpy()
        )

    return total_error / total_values


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("\nAQI SRGAN V2 TRAINING\n")
    print("---------------------")

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    print("Loading data...")

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    ).astype(np.float32)

    y_train = np.load(
        DATA_DIR / "y_train.npy"
    ).astype(np.float32)

    X_validation = np.load(
        DATA_DIR / "X_validation.npy"
    ).astype(np.float32)

    y_validation = np.load(
        DATA_DIR / "y_validation.npy"
    ).astype(np.float32)

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
    # Datasets
    # -----------------------------------------------------

    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (X_train, y_train)
        )
        .shuffle(
            buffer_size=min(
                len(X_train),
                4096
            ),
            seed=SEED,
            reshuffle_each_iteration=True
        )
        .batch(
            BATCH_SIZE
        )
        .prefetch(
            tf.data.AUTOTUNE
        )
    )

    validation_dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                X_validation,
                y_validation
            )
        )
        .batch(
            BATCH_SIZE
        )
        .prefetch(
            tf.data.AUTOTUNE
        )
    )

    # -----------------------------------------------------
    # Load V2 generator
    # -----------------------------------------------------

    print(
        "\nLoading pretrained "
        "Generator V2..."
    )

    generator = tf.keras.models.load_model(
        PRETRAINED_GENERATOR,
        custom_objects={
            "BicubicResize": BicubicResize,
            "ClipAQI": ClipAQI
        },
        compile=False
    )

    print(
        "Generator loaded successfully."
    )

    # -----------------------------------------------------
    # Build discriminator
    # -----------------------------------------------------

    print(
        "Building discriminator..."
    )

    discriminator = build_discriminator()

    print(
        "Discriminator parameters:",
        discriminator.count_params()
    )

    # -----------------------------------------------------
    # Optimizers
    # -----------------------------------------------------

    generator_optimizer = (
        tf.keras.optimizers.Adam(
            learning_rate=GENERATOR_LR,
            beta_1=0.9,
            beta_2=0.999
        )
    )

    discriminator_optimizer = (
        tf.keras.optimizers.Adam(
            learning_rate=DISCRIMINATOR_LR,
            beta_1=0.9,
            beta_2=0.999
        )
    )

    binary_crossentropy = (
        tf.keras.losses.BinaryCrossentropy(
            from_logits=True
        )
    )

    # -----------------------------------------------------
    # One GAN training step
    # -----------------------------------------------------

    @tf.function
    def train_step(
        low_resolution,
        high_resolution
    ):

        # -----------------------------------------------
        # 1. Train discriminator
        # -----------------------------------------------

        with tf.GradientTape() as discriminator_tape:

            generated = generator(
                low_resolution,
                training=False
            )

            real_logits = discriminator(
                high_resolution,
                training=True
            )

            fake_logits = discriminator(
                tf.stop_gradient(
                    generated
                ),
                training=True
            )

            # Mild label smoothing
            real_labels = (
                tf.ones_like(
                    real_logits
                ) * 0.9
            )

            fake_labels = (
                tf.zeros_like(
                    fake_logits
                )
            )

            real_loss = binary_crossentropy(
                real_labels,
                real_logits
            )

            fake_loss = binary_crossentropy(
                fake_labels,
                fake_logits
            )

            discriminator_loss = (
                real_loss + fake_loss
            ) / 2.0

        discriminator_gradients = (
            discriminator_tape.gradient(
                discriminator_loss,
                discriminator.trainable_variables
            )
        )

        discriminator_optimizer.apply_gradients(
            zip(
                discriminator_gradients,
                discriminator.trainable_variables
            )
        )

        # -----------------------------------------------
        # 2. Train generator
        # -----------------------------------------------

        with tf.GradientTape() as generator_tape:

            generated = generator(
                low_resolution,
                training=True
            )

            generated_logits = discriminator(
                generated,
                training=False
            )

            # Reconstruction / content loss
            content_loss = tf.reduce_mean(
                tf.abs(
                    high_resolution
                    - generated
                )
            )

            # Generator wants discriminator
            # to classify generated maps as real
            adversarial_loss = (
                binary_crossentropy(
                    tf.ones_like(
                        generated_logits
                    ),
                    generated_logits
                )
            )

            total_generator_loss = (
                content_loss
                + ADVERSARIAL_WEIGHT
                * adversarial_loss
            )

        generator_gradients = (
            generator_tape.gradient(
                total_generator_loss,
                generator.trainable_variables
            )
        )

        generator_optimizer.apply_gradients(
            zip(
                generator_gradients,
                generator.trainable_variables
            )
        )

        return (
            content_loss,
            adversarial_loss,
            total_generator_loss,
            discriminator_loss
        )

    # -----------------------------------------------------
    # Initial validation
    # -----------------------------------------------------

    initial_validation_mae = (
        calculate_validation_mae(
            generator,
            validation_dataset
        )
    )

    print(
        "\nInitial Generator V2 "
        f"Validation MAE: "
        f"{initial_validation_mae:.6f}"
    )

    print(
        "Approx AQI-scale MAE:",
        f"{initial_validation_mae * 250:.3f}"
    )

    # Save safe initial copy
    generator.save(
        BEST_SR_MODEL
    )

    best_validation_mae = (
        initial_validation_mae
    )

    history = []

    # -----------------------------------------------------
    # GAN training loop
    # -----------------------------------------------------

    print(
        "\nStarting adversarial training..."
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Adversarial weight: "
        f"{ADVERSARIAL_WEIGHT}\n"
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        start_time = time.time()

        content_values = []
        adversarial_values = []
        generator_values = []
        discriminator_values = []

        for (
            low_resolution,
            high_resolution
        ) in train_dataset:

            (
                content_loss,
                adversarial_loss,
                generator_loss,
                discriminator_loss
            ) = train_step(
                low_resolution,
                high_resolution
            )

            content_values.append(
                float(
                    content_loss.numpy()
                )
            )

            adversarial_values.append(
                float(
                    adversarial_loss.numpy()
                )
            )

            generator_values.append(
                float(
                    generator_loss.numpy()
                )
            )

            discriminator_values.append(
                float(
                    discriminator_loss.numpy()
                )
            )

        validation_mae = (
            calculate_validation_mae(
                generator,
                validation_dataset
            )
        )

        elapsed = (
            time.time() - start_time
        )

        mean_content = np.mean(
            content_values
        )

        mean_adversarial = np.mean(
            adversarial_values
        )

        mean_generator = np.mean(
            generator_values
        )

        mean_discriminator = np.mean(
            discriminator_values
        )

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"  Content MAE       : "
            f"{mean_content:.6f}"
        )

        print(
            f"  Adversarial Loss  : "
            f"{mean_adversarial:.6f}"
        )

        print(
            f"  Generator Loss    : "
            f"{mean_generator:.6f}"
        )

        print(
            f"  Discriminator Loss: "
            f"{mean_discriminator:.6f}"
        )

        print(
            f"  Validation MAE    : "
            f"{validation_mae:.6f}"
        )

        print(
            f"  AQI-scale Val MAE : "
            f"{validation_mae * 250:.3f}"
        )

        print(
            f"  Time              : "
            f"{elapsed:.1f} sec"
        )

        # -------------------------------------------------
        # Save best validation generator
        # -------------------------------------------------

        if (
            validation_mae
            < best_validation_mae
        ):

            print(
                "  ✅ Validation improved."
            )

            best_validation_mae = (
                validation_mae
            )

            generator.save(
                BEST_SR_MODEL
            )

            print(
                "  Best SRGAN generator saved."
            )

        history.append(
            {
                "Epoch": epoch,
                "Content_MAE": mean_content,
                "Adversarial_Loss":
                    mean_adversarial,
                "Generator_Loss":
                    mean_generator,
                "Discriminator_Loss":
                    mean_discriminator,
                "Validation_MAE_Normalized":
                    validation_mae,
                "Validation_MAE_AQI":
                    validation_mae * 250,
                "Time_Seconds":
                    elapsed,
            }
        )

        print()

    # -----------------------------------------------------
    # Save final adversarial models
    # -----------------------------------------------------

    generator.save(
        FINAL_SR_MODEL
    )

    discriminator.save(
        DISCRIMINATOR_MODEL
    )

    # -----------------------------------------------------
    # Save training history
    # -----------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_path = (
        OUTPUT_DIR
        / "srgan_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    final_validation_mae = (
        calculate_validation_mae(
            generator,
            validation_dataset
        )
    )

    print(
        "\nSRGAN TRAINING COMPLETE"
    )

    print(
        "-----------------------"
    )

    print(
        f"Initial Validation MAE : "
        f"{initial_validation_mae:.6f}"
    )

    print(
        f"Best Validation MAE    : "
        f"{best_validation_mae:.6f}"
    )

    print(
        f"Final Validation MAE   : "
        f"{final_validation_mae:.6f}"
    )

    print(
        "\nBest validation model:"
    )

    print(
        BEST_SR_MODEL
    )

    print(
        "\nFinal adversarial generator:"
    )

    print(
        FINAL_SR_MODEL
    )

    print(
        "\nDiscriminator:"
    )

    print(
        DISCRIMINATOR_MODEL
    )

    print(
        "\nTraining history:"
    )

    print(
        history_path
    )

    print(
        "\nIMPORTANT:"
        "\nThis is an SRGAN-style AQI "
        "super-resolution prototype trained "
        "on interpolated modelled AQI surfaces, "
        "not street-level sensor ground truth."
    )


if __name__ == "__main__":
    main()