import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import wandb
from wandb.keras import WandbCallback

from models.model import load_model
from data.mnist import load
from config.hparams import HP_LEARNING_RATE, METRIC_ACCURACY

X_train, Y_train, X_test, Y_test = load()


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)

        wandb.init(
            project="my-test-project",
            config={
                "learning_rate": hparams[HP_LEARNING_RATE]
            }
        )

        model = load_model(hparams)
        model.fit(X_train, Y_train, epochs=5,
                  batch_size=200, validation_split=0.1, callbacks=[WandbCallback()])
        accuracy = model.evaluate(X_test, Y_test)[1]
        y_pred = model.predict(X_test)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
        wandb.finish()
