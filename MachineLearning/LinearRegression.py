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
def predict (new_radio,weight,bias) :
    return weight*new_radio+bias
def loss_function(x,y,weight,bias):
    n = len(x)
    sum_error=0
    for i in range(n) :
        sum_error += 0.5*(y[i]-(weight*x[i]+bias))**2
    return sum_error/n
def update_weight (x,y,weight,bias,learning_rate) :
    n = len(x)
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range (n) :
        weight_temp+=-x[i]*(y[i]-(x[i]*weight+bias))
        bias_temp+=-(y[i]-(x[i]*weight+bias))
    weight -= (weight_temp/n)*learning_rate
    bias -= (bias/n)*learning_rate
    return weight,bias
def train (x,y,weight,bias,learning_rate, iter) :
    loss_history =[]
    for i in range (iter) :
        weight,bias = update_weight(x,y,weight,bias, learning_rate)
        loss = loss_function(x,y,weight,bias)
        loss_history.append(loss)
    return weight,bias,loss_history
df = pd.read_csv("Advertising.csv",header = 0);
print(df)
x_test = df.values[:,2];
y_test = df.values[:,4];
weight,bias,loss = train(x_test,y_test,0.03,0.0014,0.0001,30)
print("result weight: {}\n result bias {}".format(weight,bias))
for i in loss :
    print("Loss : {} ".format(i))
print("Predict : {}".format(predict(19,weight,bias)))
plt.plot(x_test,y_test,"o");
plt.show();
yTime_series = np.array([i for i in range(30)])
plt.plot(yTime_series,loss)
plt.show()