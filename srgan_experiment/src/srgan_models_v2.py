import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    PReLU,
    Add,
)


# ---------------------------------------------------------
# Bicubic Upscaling Layer
# 8x8 -> 32x32
# ---------------------------------------------------------

class BicubicResize(tf.keras.layers.Layer):

    def __init__(self, target_size=(32, 32), **kwargs):
        super().__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(
            inputs,
            size=self.target_size,
            method="bicubic"
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "target_size": self.target_size
        })
        return config


# ---------------------------------------------------------
# Clip Output to Normalized AQI Range [-1, 1]
# ---------------------------------------------------------

class ClipAQI(tf.keras.layers.Layer):

    def call(self, inputs):
        return tf.clip_by_value(
            inputs,
            -1.0,
            1.0
        )


# ---------------------------------------------------------
# Residual Correction Block
# ---------------------------------------------------------

def correction_block(x, filters=64):

    shortcut = x

    x = Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = Conv2D(
        filters,
        kernel_size=3,
        padding="same"
    )(x)

    x = Add()([
        shortcut,
        x
    ])

    return x


# ---------------------------------------------------------
# Generator V2
# Bicubic + Learned Residual Correction
# ---------------------------------------------------------

def build_generator_v2():

    inputs = Input(
        shape=(8, 8, 1),
        name="low_resolution_aqi"
    )

    # -----------------------------------------------------
    # Strong baseline
    # -----------------------------------------------------

    bicubic = BicubicResize(
        target_size=(32, 32),
        name="bicubic_upscale"
    )(inputs)

    # -----------------------------------------------------
    # Learn correction from bicubic image
    # -----------------------------------------------------

    x = Conv2D(
        64,
        kernel_size=5,
        padding="same"
    )(bicubic)

    x = PReLU(
        shared_axes=[1, 2]
    )(x)

    long_skip = x

    # Residual correction blocks
    for _ in range(6):
        x = correction_block(
            x,
            filters=64
        )

    x = Conv2D(
        64,
        kernel_size=3,
        padding="same"
    )(x)

    x = Add()([
        long_skip,
        x
    ])

    # -----------------------------------------------------
    # Predict only the correction
    # -----------------------------------------------------

    correction = Conv2D(
        1,
        kernel_size=3,
        padding="same",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="aqi_correction"
    )(x)

    # Starting output = exactly bicubic
    output = Add(
        name="bicubic_plus_correction"
    )([
        bicubic,
        correction
    ])

    output = ClipAQI(
        name="final_aqi_clip"
    )(output)

    return Model(
        inputs,
        output,
        name="AQI_SR_Generator_V2"
    )


# ---------------------------------------------------------
# Architecture Test
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\nBUILDING GENERATOR V2\n")

    generator = build_generator_v2()

    generator.summary()

    print(
        "\nGenerator input:",
        generator.input_shape
    )

    print(
        "Generator output:",
        generator.output_shape
    )

    # -----------------------------------------------------
    # Verify initial model equals bicubic baseline
    # -----------------------------------------------------

    test_input = tf.random.uniform(
        shape=(1, 8, 8, 1),
        minval=-1.0,
        maxval=1.0
    )

    model_output = generator(
        test_input,
        training=False
    )

    bicubic_output = tf.image.resize(
        test_input,
        size=(32, 32),
        method="bicubic"
    )

    bicubic_output = tf.clip_by_value(
        bicubic_output,
        -1.0,
        1.0
    )

    difference = tf.reduce_max(
        tf.abs(
            model_output - bicubic_output
        )
    )

    print(
        "\nInitial max difference "
        "from bicubic:",
        float(difference)
    )

    print(
        "\nGENERATOR V2 ARCHITECTURE CHECK COMPLETE"
    )