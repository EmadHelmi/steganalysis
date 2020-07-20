import os
import keras
import numpy as np
import argparse
import matplotlib.pyplot as plt
import datetime

from PIL import Image

from models.model_v1_0_0 import build_model

parser = argparse.ArgumentParser(
    description='Train Catalyst Kernels algorithm on some dataset')
parser.add_argument("--ctrp", help="Cover Train Path (default=dataset/train/cover)",
                    type=str, default="dataset/train/cover")
parser.add_argument("--ctep", help="Cover Test Path (default=dataset/test/cover)",
                    type=str, default="dataset/test/cover")
parser.add_argument("--strp", help="Stego Train Path (default=dataset/train/stego)",
                    type=str, default="dataset/train/stego")
parser.add_argument("--step", help="Stego Test Path (default=dataset/test/stego)",
                    type=str, default="dataset/test/stego")
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

parser.add_argument("-v", default=1,
                    help="Verbosity level (0: silent, 1: complete, 2: compact, default=1)")


class ModelSaver(keras.callbacks.Callback):
    """
    This class is used for saving model in specific epochs.

    Attributes:
        saved_models_path (str): The path which models will be saved in.
    """

    def __init__(self, saved_models_path):
        """
        The constructor for ModelSaver class.

        Parameters:
            saved_models_path (str): The path which models will be saved in.
        """
        keras.callbacks.Callback.__init__(self)
        self.out_dir = saved_models_path
        self.subdir_name = datetime.datetime.now().date().strftime("%Y_%m_%d")
        self.make_out_dir()

    def make_out_dir(self):
        """
        The function to make the output directory.
        """
        if not os.path.isdir(self.out_dir + "/" + self.subdir_name):
            print("Output directory does not exists, so we create it")
            os.makedirs(self.out_dir + "/" + self.subdir_name)

    def on_epoch_end(self, epoch, logs={}):
        """
        The function which will be run on each epoch end.

        Parameters:
            epoch (int): The epoch number.
            logs (dict): Will be more information about the epoch.
        """
        if ((epoch+1) and ((epoch+1) % 10 == 0)):
            self.model.save("%s/%s/checkpoint_%s.hd5" %
                            (self.out_dir, self.subdir_name, epoch))
            print("Model saved in %s/%s as checkpoint_%s.hd5" %
                  (self.out_dir, self.subdir_name, (epoch+1)))


def load_data(
        cover_trainset_path,
        cover_testset_path,
        stego_trainset_path,
        stego_testset_path):
    """
    The function will load data based on paths.

    Parameters:
        cover_trainset_path (str): Path of cover train set images
        cover_testset_path (str): Path of cover test set images
        stego_trainset_path (str): Path of stego train set images
        stego_testset_path (str): Path of stego test set images

    Returns:
        (numpy.array, numpy.array): (train images, train labels)
        (numpy.array, numpy.array): (test images, test labels)
    """
    # Load train data cover images
    x_train_cover = np.array(
        [
            np.array(
                Image.open(cover_trainset_path + "/" + fname)
            )
            for fname in os.listdir(cover_trainset_path)
        ])
    # Set train data cover image labels
    y_train_cover = [
        0 for i in range(len(os.listdir(cover_trainset_path)))
    ]

    # Load train data stego images
    x_train_stego = np.array(
        [
            np.array(
                Image.open(stego_trainset_path + "/" + fname)
            )
            for fname in os.listdir(stego_trainset_path)
        ])
    # Set train data stego image labels
    y_train_stego = [
        1 for i in range(len(os.listdir(stego_trainset_path)))
    ]

    # Load test data cover images
    x_test_cover = np.array(
        [
            np.array(
                Image.open(cover_testset_path + "/" + fname)
            )
            for fname in os.listdir(cover_testset_path)
        ])
    # Set test data cover image labels
    y_test_cover = [
        0 for i in range(len(os.listdir(cover_testset_path)))
    ]

    # Load test data stego images
    x_test_stego = np.array(
        [
            np.array(
                Image.open(stego_testset_path + "/" + fname)
            )
            for fname in os.listdir(stego_testset_path)
        ]
    )

    # Set test data stego image labels
    y_test_stego = [
        1 for i in range(len(os.listdir(stego_testset_path)))
    ]

    # Reshape train and test data to be in (n,row,col,chan) format
    # NOTE: If your tensor backend set to be channel first, please edit these lines
    x_train_cover = x_train_cover.reshape(
        (
            x_train_cover.shape[0],
            x_train_cover.shape[1],
            x_train_cover.shape[2],
            1
        ))
    x_train_stego = x_train_stego.reshape(
        (
            x_train_stego.shape[0],
            x_train_stego.shape[1],
            x_train_stego.shape[2],
            1
        ))
    x_test_cover = x_test_cover.reshape(
        (
            x_test_cover.shape[0],
            x_test_cover.shape[1],
            x_test_cover.shape[2],
            1
        ))
    x_test_stego = x_test_stego.reshape(
        (
            x_test_stego.shape[0],
            x_test_stego.shape[1],
            x_test_stego.shape[2],
            1
        ))

    # Concat datasets
    x_train = np.concatenate(
        (x_train_cover, x_train_stego),
        axis=0)
    y_train = np.concatenate(
        (np.array(y_train_cover), np.array(y_train_stego)),
        axis=0)

    x_test = np.concatenate(
        (x_test_cover, x_test_stego),
        axis=0)
    y_test = np.concatenate(
        (np.array(y_test_cover), np.array(y_test_stego)),
        axis=0)
    return (x_train, y_train), (x_test, y_test)


def assert_model(input_shape=(1, 128, 128, 1)):
    """
    The function to assert built model with a dummy input.

    This function will raise an error when it has any problem with built model.

    Parameters:
        input_shape (tuple): Tuple of the input of the model. It should be in the form of (1, ..., ...).
    """
    test_image = np.random.rand(*input_shape)
    model = build_model(input_shape)
    model.summary()
    out = model.predict(test_image)
    assert out.shape == (1, 2)
    print(out)
    print("Model is correct")


def prepare_data(
    cover_trainset_path,
    cover_testset_path,
    stego_trainset_path,
    stego_testset_path,
    shuffle=True
):
    """
    The function will prepare data based on input arguments.

    Parameters:
        cover_trainset_path (str): Path of cover train set images.
        cover_testset_path (str): Path of cover test set images.
        stego_trainset_path (str): Path of stego train set images.
        stego_testset_path (str): Path of stego test set images.
        shuffle (bool): Wether to shuffle data.

    Returns:
        (numpy.array, numpy.array): (train images, train labels).
        (numpy.array, numpy.array): (test images, test labels).

    """
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_data(
        cover_trainset_path,
        cover_testset_path,
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


def plot_results(results, epochs, out_dir):
    """
    The function to show results on each epoch.

    Parameters:
        results (keras.history): History of each epoch. It comes directly from keras.
        epochs (int): The number of epochs.
    """
    _, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Losses")
    ax1.plot(
        range(1, epochs+1),
        results.history['val_loss'],
        label="Validation loss",
        marker='o')
    ax1.plot(
        range(1, epochs+1),
        results.history['loss'],
        label="loss",
        marker='o')
    ax1.legend()

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracies")
    ax2.plot(
        range(1, epochs+1),
        [accuracy * 100 for accuracy in results.history['accuracy']],
        label="Accuracy",
        marker='o')
    ax2.plot(
        range(1, epochs+1),
        [accuracy * 100 for accuracy in results.history['val_accuracy']],
        label="validation accuracy",
        marker='o')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        "%s/%s/result.png" % (
            out_dir,
            datetime.datetime.now().date().strftime("%Y_%m_%d")
        ),
        format="png",
        papertype="letter",
        pad_inches=0.5,
        dpi=1000
    )
    plt.show()
