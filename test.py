import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
import numpy as np
import pandas as pd
from keras.utils import to_categorical
import cv2
import tensorflow as tf
from classification_evaluator import ContinualClassifierEvaluator
from utils import divide_dataset_into_tasks,split_train_test,get_permute_mnist_tasks
from EWC_classifier import EWCClassifier
from keras.datasets import mnist
import os

task ='permnist'


if task is 'mnist':
    #divided mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = x_train.reshape(60000,784)/255.0
    tasks, labels = divide_dataset_into_tasks(X,y_train,5)
    
if task is 'permnist':
    tasks, labels = get_permute_mnist_tasks(5,1250)

tasks, labels, test_tasks, test_labels = split_train_test(tasks,labels)
ewc = EWCClassifier((tasks[0].shape[1],),fisher_n=3000,epochs=5,batch=20,ewc_lambda=50,lr=0.01,optimizer='sgd',model={'layers':2, 'units':100,'dropout':0,'activation':'relu'})
evaluator = ContinualClassifierEvaluator(ewc, tasks, labels, test_tasks, test_labels)
evaluator.train(verbose=1)
evaluator.evaluate(save_accuracies_to_file='accuracies.npy')
evaluator.evaluate(True,save_accuracies_to_file='test_accuracies.npy',)