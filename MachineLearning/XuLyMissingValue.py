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
data_frame = pd.read_csv("data5.csv", header= None)
print(data_frame)
x_value = data_frame.values
imp = SimpleImputer(missing_values=np.nan,strategy="mean")
# missing_values : number, string, np.nan (default) or None
#         The placeholder for the missing values. All occurrences of
#         `missing_values` will be imputed.
#
#     strategy : string, optional (default="mean")
#         The imputation strategy.
#
#         - If "mean", then replace missing values using the mean along
#           each column. Can only be used with numeric data. ## voi mean thi se phan gio trung binh chi su dung voi du lieu so
#         - If "median", then replace missing values using the median along
#           each column. Can only be used with numeric data. ## voi median thi se phan gio chon so o giua chi su dung voi du lieu so
#         - If "most_frequent", then replace missing using the most frequent
#           value along each column. Can be used with strings or numeric data. ## voi most_frequent thi phan gio voi so xuat hien nhieu nhat
#         - If "constant", then replace missing values with fill_value. Can be
#           used with strings or numeric data. ## voi constant se replace voi gia tri fill_value la 1 bien so phai them vao
#
#         .. versionadded:: 0.20
#            strategy="constant" for fixed value imputation.
#
#
## voi mos
imp.fit(x_value)
result = imp.transform(x_value)
print("Ket qua la : ")
print(result)