import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from train import run
from hparams import HP_LEARNING_RATE, HP_LOG_DIR, METRIC_ACCURACY, HP_EPOCHS


with tf.summary.create_file_writer(HP_LOG_DIR).as_default():
    hp.hparams_config(
        hparams=[HP_LEARNING_RATE, HP_EPOCHS],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')]
    )


session_num = 0

for learning_rate in HP_LEARNING_RATE.domain.values:
    for epochs in HP_EPOCHS.domain.values:
        hparams = {
            HP_LEARNING_RATE: learning_rate,
            HP_EPOCHS: epochs
        }
        run_name = f"run-{session_num}"
        print(f'--- Starting trial: {run_name}')
        print({h.name: hparams[h] for h in hparams})
        run(f'{HP_LOG_DIR}/{run_name}', hparams)
        session_num += 1
