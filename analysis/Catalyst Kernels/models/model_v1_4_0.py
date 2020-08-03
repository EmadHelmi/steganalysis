from keras.models import Model
from keras.layers import (
    Input, Conv2D, Dense,
    Flatten, BatchNormalization,
    LeakyReLU, Softmax, Dropout, convolutional,
    concatenate
)

from filters import filters


def build_model(input_shape):
    """
    The function to build the model based on 1.4.0 version.

    Parameters:
        input_shape (tuple): The input shape of the model. It should be in the form of (1, ..., ...).

    Returns:
        keras.Sequential: The built model.
    """
    input = Input(shape=input_shape[1:])

    # Conv Layer 1, Preprocessing
    layer_1_conv = Conv2D(
        4,
        kernel_size=5,
        kernel_initializer=filters,
        strides=2,
        name="Fixed_Filters",
        trainable=False
    )(input)
    layer_1_bn = BatchNormalization()(layer_1_conv)
    layer_1_output = LeakyReLU(alpha=0.1)(layer_1_bn)

    # Conv Layer 2, Feature learning
    layer_2_atrous_conv = Conv2D(
        5,
        kernel_size=5,
        dilation_rate=8
    )(layer_1_output)
    layer_2_bn_1 = BatchNormalization()(layer_2_atrous_conv)

    layer_2_conv = Conv2D(
        5,
        kernel_size=4,
        strides=2
    )(layer_1_output)
    layer_2_bn_2 = BatchNormalization()(layer_2_conv)

    cc_2 = concatenate([layer_2_bn_1, layer_2_bn_2])
    layer_2_output = LeakyReLU(alpha=0.1)(cc_2)

    # Conv Layer 3, Feature learning
    layer_3_atrous_conv = Conv2D(
        5,
        kernel_size=5,
        dilation_rate=4
    )(layer_2_output)
    layer_3_bn_1 = BatchNormalization()(layer_3_atrous_conv)

    layer_3_conv = Conv2D(
        5,
        kernel_size=4,
        strides=2
    )(layer_2_output)
    layer_3_bn_2 = BatchNormalization()(layer_3_conv)

    cc_3 = concatenate([layer_3_bn_1, layer_3_bn_2])
    layer_3_output = LeakyReLU(alpha=0.1)(cc_3)

    # Fully connected Layers, Binary classification
    fc_flatten_1 = Flatten()(layer_3_output)
    fc_dropout_1 = Dropout(0.2)(fc_flatten_1)
    fc_dense_1 = Dense(200)(fc_dropout_1)
    fc_activation_1 = LeakyReLU(alpha=0.1)(fc_dense_1)

    fc_dropout_2 = Dropout(0.2)(fc_activation_1)
    fc_dense_2 = Dense(200)(fc_dropout_2)
    fc_activation_2 = LeakyReLU(alpha=0.1)(fc_dense_2)

    fc_dropout_3 = Dropout(0.2)(fc_activation_2)
    fc_dense_3 = Dense(2)(fc_dropout_3)

    output = Softmax()(fc_dense_3)

    return Model(inputs=input, outputs=output)
