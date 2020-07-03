import os
import keras
import numpy as np
import argparse
import matplotlib.pyplot as plt
import datetime

from PIL import Image

from model import build_model

parser = argparse.ArgumentParser(
    description='Train Catalyst Kernels algorithm on some dataset')
parser.add_argument("--ctrp", help="Clean Train Path (default=clean/train)",
                    type=str, default="clean/train")
parser.add_argument("--ctep", help="Clean Test Path (default=clean/test)",
                    type=str, default="clean/test")
parser.add_argument("--strp", help="Stego Train Path (default=stego/train)",
                    type=str, default="stego/train")
parser.add_argument("--step", help="Stego Test Path (default=stego/test)",
                    type=str, default="stego/test")
parser.add_argument("--op", help="Output path for saved models",
                    type=str, default="saved_models")
parser.add_argument("--nc", type=int, default=2,
                    help="Number of classes (default=2)")
parser.add_argument("--ne", type=int, default=100,
                    help="Number of epochs (default=100)")
parser.add_argument("--bs", type=int, default=1,
                    help="Batch size (default=1)")
parser.add_argument("--assert_model", action="store_true",
                    help="Assert built model or not")
parser.add_argument("--shuffle", action="store_true",
                    help="Wether to shuffle the data or not")

parser.add_argument("-v", action='store_true',
                    help="Verbose the progress of training or not")


class ModelSaver(keras.callbacks.Callback):
    def __init__(self, saved_models_path):
        keras.callbacks.Callback.__init__(self)
        self.out_dir = saved_models_path
        self.make_out_dir()
        self.subdir_name = datetime.datetime.now().date().strftime("%Y_%m_%d")

    def make_out_dir(self):
        if not os.path.isdir(self.out_dir):
            print("Output directory does not exists, so we create it")
            os.makedirs(self.out_dir)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 == 0:
            self.model.save("%s/%s_%s.hd5" %
                            (self.out_dir, self.subdir_name, epoch))
            print("Model saved in %s as %s_%s.hd5" %
                  (self.out_dir, self.subdir_name, epoch))


def load_data(
        clean_trainset_path,
        clean_testset_path,
        stego_trainset_path,
        stego_testset_path):
    # Loading Train data
    x_train_clean = np.array([np.array(Image.open(clean_trainset_path + "/" + fname))
                              for fname in os.listdir(clean_trainset_path)])
    y_train_clean = [0 for i in range(len(os.listdir(clean_trainset_path)))]

    x_train_stego = np.array([np.array(Image.open(stego_trainset_path + "/" + fname))
                              for fname in os.listdir(stego_trainset_path)])
    y_train_stego = [1 for i in range(len(os.listdir(stego_trainset_path)))]

    # Loading Test data
    x_test_clean = np.array([np.array(Image.open(clean_testset_path + "/" + fname))
                             for fname in os.listdir(clean_testset_path)])
    y_test_clean = [0 for i in range(len(os.listdir(clean_testset_path)))]

    x_test_stego = np.array([np.array(Image.open(stego_testset_path + "/" + fname))
                             for fname in os.listdir(stego_testset_path)])
    y_test_stego = [1 for i in range(len(os.listdir(stego_testset_path)))]

    # Reshape train and test data to be in (n,row,col,chan) format
    # NOTE: If your tensor backend set to be channel first, please edit these lines
    x_train_clean = x_train_clean.reshape(
        x_train_clean.shape[0], x_train_clean.shape[1], x_train_clean.shape[2], 1)
    x_train_stego = x_train_stego.reshape(
        x_train_stego.shape[0], x_train_stego.shape[1], x_train_stego.shape[2], 1)
    x_test_clean = x_test_clean.reshape(
        x_test_clean.shape[0], x_test_clean.shape[1], x_test_clean.shape[2], 1)
    x_test_stego = x_test_stego.reshape(
        x_test_stego.shape[0], x_test_stego.shape[1], x_test_stego.shape[2], 1)

    # Concat datasets
    x_train = np.concatenate((x_train_clean, x_train_stego), axis=0)
    y_train = np.concatenate(
        (np.array(y_train_clean), np.array(y_train_stego)), axis=0)

    x_test = np.concatenate((x_test_clean, x_test_stego), axis=0)
    y_test = np.concatenate(
        (np.array(y_test_clean), np.array(y_test_stego)), axis=0)
    return (x_train, y_train), (x_test, y_test)


def assert_model(input_shape=(1, 128, 128, 1)):
    test_image = np.random.rand(*input_shape)
    model = build_model(input_shape)
    model.summary()
    out = model.predict(test_image)
    assert out.shape == (1, 2)
    print(out)
    print("Model is correct")


def prepare_data(
    clean_trainset_path,
    clean_testset_path,
    stego_trainset_path,
    stego_testset_path,
    shuffle=True
):
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_data(
        clean_trainset_path,
        clean_testset_path,
        stego_trainset_path,
        stego_testset_path
    )
    print("Data loaded successfuly")
    print("Data summary")
    print("Train:", (x_train.shape, y_train.shape),
          "Test:", (x_test.shape, y_test.shape))
    if shuffle:
        shuffle_map = np.arange(x_train.shape[0])
        np.random.shuffle(shuffle_map)
        x_train = x_train[shuffle_map]
        y_train = y_train[shuffle_map]

        shuffle_map = np.arange(x_test.shape[0])
        np.random.shuffle(shuffle_map)
        x_test = x_test[shuffle_map]
        y_test = y_test[shuffle_map]

    return (x_train, y_train), (x_test, y_test)


def plot_results(results, epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("losses")
    ax1.plot(range(1, epochs+1),
             results.history['val_loss'], label="validation loss", marker='o')
    ax1.plot(range(1, epochs+1),
             results.history['loss'], label="loss", marker='o')

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("accuracies")
    ax2.plot(range(1, epochs+1),
             results.history['accuracy'], label="accuracy", marker='o')
    ax2.plot(range(1, epochs+1), results.history['val_accuracy'],
             label="validation accuracy", marker='o')
    fig.legend()
    plt.show()
