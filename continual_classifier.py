import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout
from AdamW import AdamW
from keras.optimizers import SGD
from abc import ABC, abstractmethod


class ContinualClassifier(ABC):
    """
    A parent class for implementations of various continual learning techniques 
    for classification. Bulids a Keras functional model (or accepts a premade 
    one). 
    """
    
    """
    Normal init stuff. If singleheaded_classes is not None, it should be the
    total number of classes across all tasks. If model's not a dict with
    architecture specs, it should be a keras model (with appropriately-named
    layers). Possible optimizers are 'sgd' and 'adam', although adam is actually
    AdamW. Any loss and any activation from keras api can be used.
    """
    def __init__(self, shape, optimizer='adam', lr=0.00001, epochs=10, loss='categorical_crossentropy', metrics=['accuracy'], singleheaded_classes=None, model={'layers':3, 'units':400,'dropout':0,'activation':'relu'}):
        self.epochs = epochs
        if isinstance(model,dict):
            inp = Input(shape,name='inputlayer')
            x = inp
            for i in range(0,model['layers']):
                x = Dense(model['units'],model['activation'],name='dense%d'%i)(x)
                if model[dropout]>0:
                    x = Dropout(model[dropout])(x)
            if singleheaded_classes_classes is not None:
                x = Dense(singleheaded_classes,'softmax',name='singlehead')(x)
            self.model=Model(inp,x)
            optim = AdamW(lr)
            if optimizer is not 'adam':
                optim = SGD(lr)
            self.model.compile(loss=loss,optimizer=optim,metrics=metrics)
        else:
            self.model=model
        
        
    
    @abstractmethod
    def task_fit(self, X, Y, validation_data=None, verbose=0):
        pass
    
    @abstractmethod
    def save_model(filename):
        pass
        
    @abstractmethod
    def load_model(filename):
        pass