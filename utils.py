import numpy as np

from keras.datasets import cifar10
from keras.utils import to_categorical

def prepare_train_test():
    # load
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # standardization
    x_train /= 255
    x_test /= 255
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    return x_train, x_test, y_train_cat, y_test_cat, x_train_mean
