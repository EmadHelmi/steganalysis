import keras
import numpy as np
import datetime

from keras.losses import BinaryCrossentropy
from keras.callbacks import LambdaCallback
from keras.optimizers import RMSprop
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from model import build_model
from utils import parser, prepare_data, assert_model, plot_results, ModelSaver


def train(
        train_sets: tuple,
        test_sets: tuple,
        input_shape: tuple = (1, 128, 128, 1),
        epochs: int = 100,
        classes: int = 2,
        batch_size: int = 1,
        verbose=True,
        out_dir: str = "saved_models"):
    (x_train, y_train), (x_test, y_test) = train_sets, test_sets
    y_train = keras.utils.to_categorical(y_train, classes)
    y_test = keras.utils.to_categorical(y_test, classes)
    model = build_model(input_shape)
    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=RMSprop(learning_rate=0.0001),
        metrics=['accuracy']
    )
    saver = ModelSaver(out_dir)
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1 if verbose else 0,
        validation_data=(x_test, y_test),
        callbacks=[saver]
    )
    model.save("%s/%s/final.hd5" %
               (out_dir, datetime.datetime.now().date().strftime("%Y_%m_%d")))
    print("Model saved in %s as final.hdf5" % out_dir)
    plot_results(history, epochs)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.assert_model:
        assert_model()
    (x_train, y_train), (x_test, y_test) = prepare_data(
        args.ctrp,
        args.ctep,
        args.strp,
        args.step,
        shuffle=args.shuffle
    )
    train(
        (x_train, y_train),
        (x_test, y_test),
        input_shape=(1, *x_train.shape[1:]),
        epochs=args.ne,
        classes=args.nc,
        batch_size=args.bs,
        verbose=args.v,
        out_dir=args.op
    )
