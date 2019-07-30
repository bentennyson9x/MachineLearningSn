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
data_frame = pd.read_csv("test_file.csv", header = None)
print(data_frame)
print(data_frame[2])
exportFrame = data_frame[3]
exportFrame.to_csv("nguyendz.csv")
