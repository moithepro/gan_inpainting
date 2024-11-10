import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore

from Constants import *


def build_generator():
    """
    Builds the Generator model with named layers.
    """
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='Input')

    # Encoder
    x = layers.Conv2D(64, kernel_size=5, strides=2, padding='same', name='Conv2D_Encoder_1')(inputs)
    x = layers.LeakyReLU(name='LeakyReLU_1')(x)
    x = layers.Conv2D(128, kernel_size=5, strides=2, padding='same', name='Conv2D_Encoder_2')(x)
    x = layers.BatchNormalization(name='BatchNorm_Encoder_1')(x)
    x = layers.LeakyReLU(name='LeakyReLU_2')(x)
    x = layers.Conv2D(256, kernel_size=5, strides=2, padding='same', name='Conv2D_Encoder_3')(x)
    x = layers.BatchNormalization(name='BatchNorm_Encoder_2')(x)
    x = layers.LeakyReLU(name='LeakyReLU_3')(x)

    # Decoder
    x = layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', name='Conv2DTranspose_Decoder_1')(x)
    x = layers.BatchNormalization(name='BatchNorm_Decoder_1')(x)
    x = layers.LeakyReLU(name='LeakyReLU_4')(x)
    x = layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', name='Conv2DTranspose_Decoder_2')(x)
    x = layers.BatchNormalization(name='BatchNorm_Decoder_2')(x)
    x = layers.LeakyReLU(name='LeakyReLU_5')(x)
    outputs = layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh', name='Output')(x)

    model = models.Model(inputs, outputs, name='Generator')
    return model


def build_discriminator():
    """
    Builds the Discriminator model with named layers.
    """
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='Input')
    x = layers.Conv2D(64, kernel_size=5, strides=2, padding='same', name='Conv2D_1')(inputs)
    x = layers.LeakyReLU(0.2, name='LeakyReLU_1')(x)
    x = layers.Conv2D(128, kernel_size=5, strides=2, padding='same', name='Conv2D_2')(x)
    x = layers.BatchNormalization(name='BatchNorm_1')(x)
    x = layers.LeakyReLU(0.2, name='LeakyReLU_2')(x)
    x = layers.Conv2D(256, kernel_size=5, strides=2, padding='same', name='Conv2D_3')(x)
    x = layers.BatchNormalization(name='BatchNorm_2')(x)
    x = layers.LeakyReLU(0.2, name='LeakyReLU_3')(x)
    x = layers.Flatten(name='Flatten')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='Output')(x)

    model = models.Model(inputs, outputs, name='Discriminator')
    return model
