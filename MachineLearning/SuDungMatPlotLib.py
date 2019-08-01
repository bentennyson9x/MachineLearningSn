import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import axes3d
from sklearn import tree
from mpl_toolkits.mplot3d import axes3d
import cv2
x=np.array([3,5])
y=np.array([7,9])
plt.plot(x,y,"o")
#plt.show()
img= cv2.imread("TT.JPG",cv2.IMREAD_COLOR)
cv2.imshow('TrangOcCho',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# plt.imshow(img)
# plt.show()