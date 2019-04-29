import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
import numpy as np
import pandas as pd
from keras.utils import to_categorical
import cv2
import tensorflow as tf
from evaluate import ContinualClassifierEvaluator, divide_dataset_into_tasks
from EWC_classifier import EWCClassifier
from keras.datasets import mnist
import os


#df = pd.read_csv('merged1000')
#df = df.drop(['Unnamed: 0'], axis=1)
#labels = df['1']
#Y = np.argmax(pd.get_dummies(labels).values, axis=1)
#df = df.drop(['1'], axis=1)
#X = np.array(df)
#tasks, labels = divide_dataset_into_tasks(X,Y,20)

#X = np.random.rand(100,20)
#Y = np.array([0]*10 + [1]*10 +[2]*10 + [3]*10 + [4]*10 + [5]*10 + [6]*10 + [7]*10 + [8]*10 + [9]*10)
#tasks, labels = divide_dataset_into_tasks(X,Y,5)




import pandas as pd
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os, cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split




def get_hindi_tasks(path = 'vowels'): # vowels or numerals
    # enumerating labels, original values can be retrieved by indexing the 'labels' array

    if path == "vowels":
        labels = ['a', 'aa', 'i', 'ee',  'u', 'oo', 'ae', 'ai', 'o', 'au', 'an', 'ah']

    num_labels = []
    for num, val in enumerate(labels):
      num_labels.append(num)
      
    folders = [i+1 for i in num_labels] # adding ones, since folders are arranged from 1-12

    X = []
    y = []

    for f in folders:
      abs_path = os.path.join(path, str(f))
      for img in os.listdir(abs_path):
        X.append(cv2.imread(os.path.join(abs_path, img), cv2.IMREAD_GRAYSCALE))
        y.append(f - 1) # converting fodler value to original label val


    X = np.array(X)
    y = np.array(y)

    

    #X = np.split(X, 4)
    #y = to_categorical(y)
    #y = np.split(y, 4)

    # normalizing the input data modify if necessary
    X = tf.keras.utils.normalize(X, axis = 1)
    return X, y



X, y_train = get_hindi_tasks()
X = X.reshape([X.shape[0], X.shape[1]**2])
tasks, labels = divide_dataset_into_tasks(X,y_train,6)


print(X.shape)
ewc = EWCClassifier((X.shape[1],),fisher_n=3000,epochs=10,ewc_lambda=200, lr=0.00005)
evaluator = ContinualClassifierEvaluator(ewc, tasks, labels)
evaluator.train()
evaluator.evaluate()