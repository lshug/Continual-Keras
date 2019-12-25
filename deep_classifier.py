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
    def __init__(self,  *args, **kwargs):        
        super().__init__(*args, **kwargs)
    
    def task_fit_method(self, X, Y, model, new_task, batch_size, epochs, validation_data=None, verbose=2):
        i = 0
        j = 0
        model.compile(loss=self.loss,optimizer=self.optimizer,metrics=['accuracy'])
        model.fit(X,Y, batch_size=batch_size, epochs=epochs, validation_data = validation_data, verbose=verbose, shuffle=True)
