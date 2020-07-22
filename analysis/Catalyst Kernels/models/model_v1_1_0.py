from keras.models import Sequential
from keras.layers import (
    Conv2D, Dense,
    Flatten, BatchNormalization, LeakyReLU, Softmax, Dropout
)

from filters import filters


def build_model(input_shape):
    """
    The function to build the model based on any architecture.

    Parameters:
        input_shape (tuple): The input shape of the model. It should be in the form of (1, ..., ...).

    Returns:
        keras.Sequential: The built model.
    """
    model = Sequential()
    # Conv Layer 1, Preprocessing
    model.add(Conv2D(
        4,
        kernel_size=5,
        kernel_initializer=filters,
        input_shape=input_shape[1:],
        strides=2,
        name="Fixed_Filters",
        trainable=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    # Conv Layer 2, Feature learning
    model.add(Conv2D(
        5,
        kernel_size=5,
        strides=2
    ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    # Conv Layer 3, Feature learning
    model.add(Conv2D(
        10,
        kernel_size=5,
        strides=2
    ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    # Fully connected Layers, Binary classification
    model.add(Flatten())
    model.add(Dense(200))

    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(200))

    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(2))
    model.add(Softmax())

    return model
