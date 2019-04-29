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



(x_train, y_train), (x_test, y_test) = mnist.load_data()
X = x_train.reshape(60000,784)/255.0
tasks, labels = divide_dataset_into_tasks(X,y_train,5)


ewc = EWCClassifier((X.shape[1],),fisher_n=3000,epochs=10,ewc_lambda=500,lr=0.00005,empirical=True)
evaluator = ContinualClassifierEvaluator(ewc, tasks, labels)
evaluator.train()
evaluator.evaluate()