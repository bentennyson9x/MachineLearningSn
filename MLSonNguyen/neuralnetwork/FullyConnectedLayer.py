import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.utils.tests.test_wildcard import obj_t
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import axes3d
from sklearn import tree
from mpl_toolkits.mplot3d import axes3d
import cv2
from sklearn.model_selection import train_test_split  ## dung de tach bo test ra
from sklearn.datasets import load_iris
## ham train_test_split se tach bo train voi bo test ra rieng biet de xay dung model voi nhung thu vien dataset
## ham train_test_split se tra ve 4 doi so,  doi so 1 danh cho x_train dung de trainning voi doi so thu 3 y_train
## 2 doi so con lai tuong tu nhung dung de test
from neuralnetwork.Layer import Layer
class FullyConnectedLayer (Layer) :
    def __init__(self, input_shape, output_shape):
        """
        (1,3)(3,4)=(1,4)
        (3,1)(1,4)=(3,4)
        weights = (3,4)
        :param input_shape: (1,3)
        :param output_shape: (1,4)
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.rand(input_shape[1], output_shape[1]) - 0.5
        self.bias = np.random.rand(1, output_shape[1]) - 0.5
    def forward_propagation(self, input):
        """

        :param input: (1,3) (3,4) = (1,4)
        :return:
        """
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        """

        :param output_error = ouput_shape:  (1,4) (4,3) = (1,3)
        :param learning_rate:
        :return:
        """
        curent_layer_err = np.dot(output_error, self.weights.T)
        dweight = np.dot(self.input.T, output_error)

        self.weights -= dweight * learning_rate
        self.bias -= learning_rate * output_error

        return curent_layer_err
