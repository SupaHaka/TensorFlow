import tensorflow as tf


class TensStuff:

    def evaluate(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                            tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                            tf.keras.layers.Dropout(0.2),
                                            tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5)

        return model.evaluate(x_test, y_test)

    def fit(self):
        inputs = tf.keras.Input(shape=(32,))  # Returns a placeholder tensor

        # A layer instance is callable on a tensor, and returns a tensor.
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

        # Instantiate the model given inputs and outputs.
        model = tf.keras.Model(inputs=inputs, outputs=predictions)

        # The compile step specifies the training configuration.
        model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Trains for 5 epochs
        # model.fit(data, labels, batch_size=32, epochs=5)

