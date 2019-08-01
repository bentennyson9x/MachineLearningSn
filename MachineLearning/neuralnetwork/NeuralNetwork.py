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
class NetWork :
    def __init__(self):
        self.layers=[]
        self.loss = None
        self.loss_derative = None
    def add (self,layer) :
        self.layers.append(layer)
    def setup_loss (self,loss,loss_derative) :
        self.loss=loss
        self.loss_derative=loss_derative
    def predict(self,inputs):
        """

        :param input: [[1,3]] => 1
        :return: ket qua du doan
        """
        result = []
        n = len(inputs)
        for i in range(n):
            output=inputs[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result
    def fit(self,x_train,y_train,learning_rate,epochs):
        n=len(x_train)
        for i in range (epochs) :
            error = 0
            for j in range(n):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                error += self.loss(y_train[j],output)
                sub_error =  self.loss_derative(y_train[j],output)
                for layer in reversed(self.layers) :
                    sub_error = layer.backward_propagation(sub_error,learning_rate)
            error = error/n
            print("epoch : %d/%d error = %f"%(i,epochs,error))
