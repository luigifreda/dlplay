# ******************************************************************************
# This file is part of dlplay
# 
# Copyright (C) Luigi Freda <luigi dot freda at gmail dot com>
# 
# dlplay is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# dlplay is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with dlplay. If not, see <http://www.gnu.org/licenses/>.
# ******************************************************************************
import tensorflow as tf
import numpy as np
import cv2


# https://www.tensorflow.org/tutorials/quickstart/beginner?hl=it
# This short introduction uses Keras to:
# 1. Load a prebuilt dataset.
# 2. Build a neural network machine learning model that classifies images.
# 3. Train this neural network.
# 4. Evaluate the accuracy of the model.

if __name__ == "__main__":

    # Load and prepare the MNIST dataset. The pixel values of the images range from 0 through 255.
    # Scale these values to a range of 0 to 1 by dividing the values by 255.0.
    # This also converts the sample data from integers to floating-point numbers:

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Sequential is useful for stacking layers where each layer has one input tensor and one
    # output tensor. Layers are functions with a known mathematical structure that can be reused
    # and have trainable variables. Most TensorFlow models are composed of layers.
    # This model uses the Flatten, Dense, and Dropout layers.
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )

    # For each example, the model returns a vector of logits or log-odds scores, one for each class.
    train_sample = x_train[:1]

    train_predictions = model(train_sample).numpy()
    print(f"train_predictions: {train_predictions}")

    # The tf.nn.softmax function converts these logits to "probabilities" for each class:
    tf.nn.softmax(train_predictions).numpy()

    # NOTE: It is possible to bake the tf.nn.softmax function into the activation function for the last
    # layer of the network. While this can make the model output more directly interpretable,
    # this approach is discouraged as it's impossible to provide an exact and numerically stable
    # loss calculation for all models when using a softmax output.

    # Define a loss function for training using losses.SparseCategoricalCrossentropy:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # The loss function takes a vector of ground truth values and a vector of logits and returns a scalar loss for each example. This loss is equal to the negative log probability of the true class: The loss is zero if the model is sure of the correct class.

    # This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.math.log(1/10) ~= 2.3.

    sample_loss = loss_fn(y_train[:1], train_predictions).numpy()
    print(f"sample_loss: {sample_loss}")

    # Before you start training, configure and compile the model using Keras Model.compile.
    # Set the optimizer class to adam, set the loss to the loss_fn function you defined earlier,
    # and specify a metric to be evaluated for the model by setting the metrics parameter to accuracy.

    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    # The Model.fit method adjusts the model parameters to minimize the loss:
    model.fit(x_train, y_train, epochs=5)

    # The Model.evaluate method checks the model's performance, usually on a validation set or test set.
    model.evaluate(x_test, y_test, verbose=2)

    # The image classifier is now trained to ~98% accuracy on this dataset. To learn more, read the TensorFlow tutorials.
    # If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    test_sample = x_test[:5]
    test_predictions = probability_model(test_sample)
    print(f"test_predictions: {test_predictions}")
