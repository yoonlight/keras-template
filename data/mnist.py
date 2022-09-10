from keras.utils.np_utils import to_categorical
from keras.datasets import mnist


def load():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(
        X_train.shape[0], 28, 28, 1).astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    return X_train, Y_train, X_test, Y_test
