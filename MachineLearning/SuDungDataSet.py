import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import axes3d
from sklearn import tree
from mpl_toolkits.mplot3d import axes3d
import cv2
from sklearn.model_selection import train_test_split ## dung de tach bo test ra
from sklearn.datasets import load_iris
## ham train_test_split se tach bo train voi bo test ra rieng biet de xay dung model voi nhung thu vien dataset
## ham train_test_split se tra ve 4 doi so,  doi so 1 danh cho x_train dung de trainning voi doi so thu 3 y_train
## 2 doi so con lai tuong tu nhung dung de test
iris_dataset = load_iris()
print(iris_dataset)
print(len(iris_dataset.target))
x_train,x_test,y_train,y_test=train_test_split(iris_dataset.data, iris_dataset.target, random_state=1)
## ham train_test_split se tach bo train voi bo test ra rieng biet de xay dung model voi nhung thu vien dataset
## ham train_test_split se tra ve 4 doi so,  doi so 1 danh cho x_train dung de trainning voi doi so thu 3 y_train
## 2 doi so con lai tuong tu nhung dung de test
print(y_test)
ID3Tree = tree.DecisionTreeClassifier()
model = ID3Tree.fit(x_train,y_train)
print(model.predict(x_test))
x_newTest = np.array([[4.5,3.2,1.7,6.1]])
print(model.predict(x_newTest))
print(model.score(x_test,y_test))
