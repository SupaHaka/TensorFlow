# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load Data
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Names of images in order of label [0-9]
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # Plot for exploration
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    # plt.show()

    # Pre-process
    # 1. Normalize pixel values
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Display some of the images with their corresponding labels
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.yticks([])
        plt.xticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])

    # plt.show()

    # Build the model
    model = keras.Sequential([
        # Re-formats the data from 2d array to a 1d array
        # Think of this layer as unstacking rows of pixels in the image and lining them up.
        keras.layers.Flatten(input_shape=(28,28)),

        # Densely/Fully connected neural network with 128 nodes
        keras.layers.Dense(128, activation=tf.nn.relu),

        # this layer is a 10-node softmax layer
        # it returns an array of 10 probability scores that sum to 1
        keras.layers.Dense(10,activation=tf.nn.softmax)
    ])

    # Compile the model
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=10)

    # Evaluate the accuracy of the model


