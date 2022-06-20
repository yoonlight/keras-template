from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizer_v2.adam import Adam

from hparams import HP_LEARNING_RATE


def load_model(hparams):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1),
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2, padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=hparams[HP_LEARNING_RATE]),
                  metrics=['accuracy'])
    return model
