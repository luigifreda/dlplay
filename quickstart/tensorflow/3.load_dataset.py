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

# from https://www.tensorflow.org/tutorials/quickstart/beginner?hl=it
if __name__ == "__main__":

    # Load and prepare the MNIST dataset. The pixel values of the images range from 0 through 255.
    # Scale these values to a range of 0 to 1 by dividing the values by 255.0.
    # This also converts the sample data from integers to floating-point numbers:

    mnist = tf.keras.datasets.mnist
    # print the dataset metadata
    print(mnist.load_data.__doc__)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
