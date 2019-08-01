import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import axes3d
from sklearn import tree
from mpl_toolkits.mplot3d import axes3d
a = np.array([1,3,4,5,3,8])
print(a)
print(a.ndim)
a = np.array ([[1,3,4,5,3,8],
               [1,3,4,5,3,8]])
print(a)
print(a.ndim)
print(a[1])
print(a[1][2:4]) ## se lay tu vi tri thu 2 den vi tri n-1 tuc 4-1 = 3
