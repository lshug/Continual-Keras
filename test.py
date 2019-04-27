import numpy as np
import pandas as pd
from evaluate import ContinualClassifierEvaluator, divide_dataset_into_tasks
from EWC_classifier import EWCClassifier
from keras.datasets import mnist


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

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000,784)/255.0
tasks, labels = divide_dataset_into_tasks(x_train,y_train,5)

ewc = EWCClassifier((X.shape[1],),fisher_n=100)
evaluator = ContinualClassifierEvaluator(ewc, tasks, labels)
evaluator.train()
evaluator.evaluate()