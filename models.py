import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

from Constants import *


def build_generator():
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # Encoder
    c1 = layers.Conv2D(64, kernel_size=5, strides=2, padding='same')(inputs)
    c1 = layers.LeakyReLU()(c1)
    c2 = layers.Conv2D(128, kernel_size=5, strides=2, padding='same')(c1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.LeakyReLU()(c2)
    c3 = layers.Conv2D(256, kernel_size=5, strides=2, padding='same')(c2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.LeakyReLU()(c3)

    # Decoder (with skip connections)
    u1 = layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(c3)
    u1 = layers.BatchNormalization()(u1)
    u1 = layers.Concatenate()([u1, c2])  # Skip connection
    u1 = layers.LeakyReLU()(u1)
    u2 = layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(u1)
    u2 = layers.BatchNormalization()(u2)
    u2 = layers.Concatenate()([u2, c1])  # Skip connection
    u2 = layers.LeakyReLU()(u2)
    outputs = layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh')(u2)

    model = models.Model(inputs, outputs, name='Generator')
    return model


def build_discriminator():
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(512, kernel_size=4, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    outputs = layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(x)  # PatchGAN output

    model = models.Model(inputs, outputs, name='Discriminator')
    return model
