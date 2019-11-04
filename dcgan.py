import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

# Generator
img_size = 64


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(img_size//8*img_size//8 *
                           256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Reshape((img_size//8, img_size//8, 256)))
    # Note: None is the batch size
    assert model.output_shape == (None, img_size//8, img_size//8, 256)

    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2DTranspose(
        128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, img_size//8, img_size//8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, img_size//4, img_size//4, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2DTranspose(
        32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, img_size//2, img_size//2, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2),
                                     padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (
        None, img_size, img_size, 3), model.output_shape

    return model

# Discriminator


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[img_size, img_size, 3]))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
