import numpy as np
from tensorflow.keras import layers, models, losses
import tensorflow as tf
import pandas as pd
import networkio as nio


def main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_val, y_val) = mnist.load_data()
    x_train, x_val = x_train / 255.0, x_val / 255.0
    x_test = read()
    x_train = np.expand_dims(x_train, axis=3)
    x_val = np.expand_dims(x_val, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    model = models.Sequential()
    model.add(layers.Conv2D(filters=6, kernel_size=5, strides=(1, 1), padding="same", activation='relu',
                            input_shape=(28, 28, 1)))
    model.add(layers.AveragePooling2D(pool_size=2))
    model.add(layers.Conv2D(filters=16, kernel_size=5, strides=(1, 1), activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=2))
    model.add(layers.Flatten())
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(120, activation='relu'))
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(84, activation='relu'))
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()

    model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=64, epochs=40, validation_data=(x_val, y_val))
    nio.save(model)


def read():
    """
    returns x_test
    """
    df = pd.read_csv("test.csv")
    x = df.to_numpy()  # read from csv and convert it to a numpy matrix
    x = x / 255  # contains x between 0 and 1, makes training easier
    m = x.shape[0]
    x = x.reshape(m, 28, 28)
    return x


if __name__ == '__main__':
    main()