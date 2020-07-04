import numpy as np
import keras.backend as K


def filters(shape, dtype=None):
    """
    The function to build first layer filters.

    This function will raise an exception when the input shape does not match maked filters.

    Parametrs:
        shape (tuple): The shape of filters.

    Returns:
        K.variable: A keras variable for the model.
    """
    f_kv = (1/12) * np.array([
        [[-1], [2], [-2], [2], [-1]],
        [[2], [-6], [8], [-6], [2]],
        [[-2], [8], [-12], [8], [-2]],
        [[2], [-6], [8], [-6], [2]],
        [[-1], [2], [-2], [2], [-1]]
    ]).astype('float32')

    f_p = (1/261) * np.array([
        [[0], [0], [5.2], [0], [0]],
        [[0], [23.4], [36.4], [23.4], [0]],
        [[5.2], [36.4], [-261], [36.4], [5.2]],
        [[0], [23.4], [36.4], [23.4], [0]],
        [[0], [0], [5.2], [0], [0]]
    ]).astype('float32')

    f_h = np.array([
        [[0.0562], [-0.1354], [0], [0.1354], [-0.0562]],
        [[0.0818], [-0.1970], [0], [0.1970], [-0.0818]],
        [[0.0926], [-0.2233], [0], [0.2233], [-0.0926]],
        [[0.0818], [-0.1970], [0], [0.1970], [-0.0818]],
        [[0.0562], [-0.1354], [0], [0.1354], [-0.0562]]
    ]).astype('float32')

    f_v = np.array([
        [[-0.0562], [-0.0818], [-0.0926], [-0.0818], [-0.0562]],
        [[0.1354], [0.1970], [0.2233], [0.1970], [0.1354]],
        [[0], [0], [0], [0], [0]],
        [[-0.1354], [-0.1970], [-0.2233], [-0.1970], [-0.1354]],
        [[0.0562], [0.0818], [0.0926], [0.0818], [0.0562]]
    ]).astype('float32')
    filters = np.ndarray((5, 5, 1, 4))
    filters[:, :, :, 0] = f_kv
    filters[:, :, :, 1] = f_p
    filters[:, :, :, 2] = f_h
    filters[:, :, :, 3] = f_v

    assert filters.shape == shape
    return K.variable(filters, dtype='float32')
