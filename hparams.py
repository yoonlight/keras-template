from tensorboard.plugins.hparams import api as hp

HP_LEARNING_RATE = hp.HParam(
    'learning_rate', hp.Discrete([1E-03, 5E-04, 1E-04, 1E-05]))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete([10]))

HP_LOG_DIR = 'logs/hparam_tuning'

METRIC_ACCURACY = 'accuracy'
