import os 
import numpy as np
import cv2 as cv

def load_data(lst_path_x, lst_path_y):
    X_train = []
    y_train = []
    for x,y in zip(lst_path_x, lst_path_y):
        X_train.append(cv.imread(x))
        y_train.append(cv.imread(y))
    return np.asarray(X_train),np.asarray(y_train)

        
    