import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from model import load_model
from dataset import load
from hparams import METRIC_ACCURACY

X_train, Y_train, X_test, Y_test = load()


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        model = load_model(hparams)
        history = model.fit(X_train, Y_train, epochs=5,
                            batch_size=200, validation_split=0.1)
        accuracy = model.evaluate(X_test, Y_test)[1]
        y_pred = model.predict(X_test)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
