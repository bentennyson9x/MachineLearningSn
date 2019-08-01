import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import axes3d
from sklearn import tree
#buoc 1 : thu thap du lieu
#buoc 2 : xu ly du lieu
#buoc 3 : trainning model
## Cannang,Chieucao,Huyetap,Vandong
featureString = np.array([["nhe","trungbinh","trungbinh","nhieu"],
                     ["nang","thap","cao","it"],
                     ["nhe","thap","cao","it"],
                     ["nang","cao","cao","tb"],
                     ["nhe","cao","cao","nhieu"],
                     ["trungbinh","thap","tb","nhieu"],
                     ["trungbinh","trungbinh","trungbinh","it"],
                     ["nang","thap","thap","nhieu"]])
##Quy uoc nhe = 1, thap = 2, trung binh = 3, cao = 4, nang = 5, ít =6, nhieu = 7
featureC = np.array([[1,3,3,7],
                     [5,2,4,6],
                     [1,2,4,6],
                     [5,4,4,3],
                     [1,4,4,7],
                     [3,2,3,7],
                     [3,3,3,6],
                     [5,2,2,7]])
##Benhtim hay khong
lable = [0,1,1,0,0,0,0,1]
DecisionTree = tree.DecisionTreeClassifier()
result = DecisionTree.fit(featureC,lable)
x_test=np.array([[5,2,2,6],
                [1,4,3,7],
                [1,2,2,6],
                 [5,3,3,3]])
predict_action = result.predict(x_test)

print(predict_action)
print(result.score(x_test,predict_action))