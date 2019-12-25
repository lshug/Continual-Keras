from continual_classifier import ContinualClassifier
from utils import estimate_fisher_diagonal
from keras.losses import categorical_crossentropy
import numpy as np
from keras.models import Model
from keras.layers import Lambda
import tensorflow as tf
import keras.backend as K
from sklearn.utils import shuffle
from tqdm import tqdm

class DeepClassifier(ContinualClassifier):
    def __init__(self,  optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'], singleheaded_classes=None, model={'layers':3, 'units':200,'dropout':0,'activation':'relu'}):        
        super().__init__(optimizer,loss,metrics,singleheaded_classes,model)
    
    def save_model(self, filename):
        pass
        
    
    def load_model(self, filename):
        pass
    
    def task_fit_method(self, X, Y, model, new_task, batch_size, epochs, validation_data=None, verbose=2):
        i = 0
        j = 0
        model.compile(loss=self.loss,optimizer=self.optimizer,metrics=['accuracy'])
        model.fit(X,Y, batch_size=batch_size, epochs=epochs, validation_data = validation_data, verbose=verbose, shuffle=True)


