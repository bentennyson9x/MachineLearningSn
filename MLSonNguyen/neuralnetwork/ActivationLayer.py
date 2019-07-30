import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from neuralnetwork import Layer
class ActivationLayer() :
    def __init__(self,input_shape,output_shape,activation,activation_derative):
        """

        :param input_shape: dau vao input la 1 mang. Vi du  (1,4)
        :param output_shape: mang
        :param activation: ham kich hoat
        :param activation_derative: dao ham ham kich hoat
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.activation_derative = activation_derative
    def forward_propagation(self, input):
        self.input=input
        self.output=self.activation(input)
        return self.output

    def backward_propagation(self,output_error, learning_rate):
        return self.activation_derative(self.input)*output_error

