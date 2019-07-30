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
from abc import abstractclassmethod
class Layer :
    def __init__(self):
        self.input = None
        self.output = None
        self.input_shape = None
        self.output_shape = None
        raise NotImplementedError
    @abstractclassmethod
    def input(self):
        return self.input
    @abstractclassmethod
    def output(self):
        return self.output
    @abstractclassmethod
    def input_shape(self):
        return self.input_shape
    @abstractclassmethod
    def output_shape (self):
        return self.output_shape
    @abstractclassmethod
    def forward_propagation(self, input):
        raise NotImplementedError
    @abstractclassmethod
    def backward_propagation(self,output_error, learning_rate):
        raise NotImplementedError


