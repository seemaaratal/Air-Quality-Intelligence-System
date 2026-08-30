import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    PReLU,
    Add,
    Dense,
    Flatten,
    LeakyReLU,
)


# ---------------------------------------------------------
# Custom Pixel Shuffle Layer
# ---------------------------------------------------------

class PixelShuffle(tf.keras.layers.Layer):

    def __init__(self, scale=2, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(
            inputs,
            block_size=self.scale
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "scale": self.scale
        })
        return config


# ---------------------------------------------------------
# Generator Residual Block
# ---------------------------------------------------------

def residual_block(x, filters=64):

    shortcut = x

    x = Conv2D(
        filters,
        kernel_size=3,
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = PReLU(
        shared_axes=[1, 2]
    )(x)

    x = Conv2D(
        filters,
        kernel_size=3,
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = Add()([
        shortcut,
        x
    ])

    return x


# ---------------------------------------------------------
# Upsampling Block
# ---------------------------------------------------------

def upsample_block(x, filters=256):

    x = Conv2D(
        filters,
        kernel_size=3,
        padding="same"
    )(x)

    x = PixelShuffle(
        scale=2
    )(x)

    x = PReLU(
        shared_axes=[1, 2]
    )(x)

    return x


# ---------------------------------------------------------
# SRGAN Generator
# 8x8 -> 32x32
# ---------------------------------------------------------

def build_generator():

    inputs = Input(
        shape=(8, 8, 1),
        name="low_resolution_aqi"
    )

    x = Conv2D(
        64,
        kernel_size=9,
        padding="same"
    )(inputs)

    x = PReLU(
        shared_axes=[1, 2]
    )(x)

    generator_skip = x

    # Residual blocks
    for _ in range(8):
        x = residual_block(
            x,
            filters=64
        )

    x = Conv2D(
        64,
        kernel_size=3,
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = Add()([
        generator_skip,
        x
    ])

    # 8x8 -> 16x16
    x = upsample_block(
        x,
        filters=256
    )

    # 16x16 -> 32x32
    x = upsample_block(
        x,
        filters=256
    )

    outputs = Conv2D(
        1,
        kernel_size=9,
        padding="same",
        activation="tanh",
        name="super_resolved_aqi"
    )(x)

    return Model(
        inputs,
        outputs,
        name="AQI_SR_Generator"
    )


# ---------------------------------------------------------
# Discriminator Block
# ---------------------------------------------------------

def discriminator_block(
    x,
    filters,
    stride,
    use_batch_norm=True
):

    x = Conv2D(
        filters,
        kernel_size=3,
        strides=stride,
        padding="same"
    )(x)

    if use_batch_norm:
        x = BatchNormalization()(x)

    x = LeakyReLU(
        negative_slope=0.2
    )(x)

    return x


# ---------------------------------------------------------
# SRGAN Discriminator
# Input: 32x32 AQI map
# ---------------------------------------------------------

def build_discriminator():

    inputs = Input(
        shape=(32, 32, 1),
        name="high_resolution_aqi"
    )

    x = discriminator_block(
        inputs,
        filters=64,
        stride=1,
        use_batch_norm=False
    )

    x = discriminator_block(
        x,
        filters=64,
        stride=2
    )

    x = discriminator_block(
        x,
        filters=128,
        stride=1
    )

    x = discriminator_block(
        x,
        filters=128,
        stride=2
    )

    x = discriminator_block(
        x,
        filters=256,
        stride=1
    )

    x = discriminator_block(
        x,
        filters=256,
        stride=2
    )

    x = discriminator_block(
        x,
        filters=512,
        stride=1
    )

    x = discriminator_block(
        x,
        filters=512,
        stride=2
    )

    x = Flatten()(x)

    x = Dense(
        1024
    )(x)

    x = LeakyReLU(
        negative_slope=0.2
    )(x)

    outputs = Dense(
        1,
        activation="sigmoid",
        name="real_or_generated"
    )(x)

    return Model(
        inputs,
        outputs,
        name="AQI_SR_Discriminator"
    )


# ---------------------------------------------------------
# Test Models
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\nBUILDING SRGAN MODELS\n")

    generator = build_generator()
    discriminator = build_discriminator()

    print("\nGENERATOR")
    print("---------")
    generator.summary()

    print("\nDISCRIMINATOR")
    print("-------------")
    discriminator.summary()

    print(
        "\nGenerator input:",
        generator.input_shape
    )

    print(
        "Generator output:",
        generator.output_shape
    )

    print(
        "\nDiscriminator input:",
        discriminator.input_shape
    )

    print(
        "Discriminator output:",
        discriminator.output_shape
    )

    print(
        "\nSRGAN MODEL ARCHITECTURE CHECK COMPLETE"
    )