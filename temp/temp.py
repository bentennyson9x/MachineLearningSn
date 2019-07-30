import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import axes3d
a1=np.array([[-3 ,0,0],
   [1,-2,1],
   [1,2,-2]])
a2=np.array([[-1,2,1],
    [0,-2,1],
    [0,2,-3]])
x1Derivative=np.zeros((100,3,3))
x2Derivative=[]
xVariable = np.arange(100)
##print(x1Derivative);
def multipleMaxtrix():
    for i in xVariable:
        x1Derivative[i]=a1*i
multipleMaxtrix();
temp= x1Derivative.flatten()
print(temp)