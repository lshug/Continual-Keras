import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
from abc import ABC, abstractmethod


class ContinualClassifier(ABC):
    """
    A parent class for implementations of various continual learning techniques 
    for classification. Bulids a Keras functional model (or accepts a premade 
    one). 
    """
    
    """
    Normal init stuff. If singleheaded_classes is not None, it should be the
    total number of classes across all tasks; in this case, the number of 
    classes in each task should be the same. If model's not a dict with
    architecture specs, it should be a keras model (with appropriately-named
    layers). Possible optimizers are 'sgd' and 'adam', although adam is actually
    AdamW. Any loss and any activation from keras api can be used.
    If singleheaded_classes is None, models are stored in self.models.
    """
    def __init__(self, shape, optimizer='adam',batch=32, lr=0.001, epochs=200, loss='categorical_crossentropy', metrics=['accuracy'], singleheaded_classes=None, model={'layers':3, 'units':400,'dropout':0,'activation':'relu'}):
        self.epochs = epochs
        self.batch=batch
        optim = Adam(lr)
        if optimizer is 'sgd':
            optim = SGD(lr)
        self.optimizer = optim
        self.loss = loss
        self.metrics = metrics
        if isinstance(model,dict):
            inp = Input(shape,name='inputlayer')
            x = inp
            for i in range(0,model['layers']):
                x = Dense(model['units'],activation=model['activation'],name='dense%d'%i)(x)
                if model['dropout']>0:
                    x = Dropout(model['dropout'])(x)
            if singleheaded_classes is not None:
                x = Dense(singleheaded_classes,activation='softmax',name='singlehead')(x)
                self.singleheaded = True
            else:
                self.singleheaded = False
                self.models = []
            self.model=Model(inp,x)
            
            self.model.compile(loss=loss,optimizer=optim,metrics=metrics)
        else:
            self.model=model
            
#    @abstractmethod
    def task_fit_method(self, X, Y, model, validation_data=None, verbose=0):
        print('No task fit method')
        
        
#    @abstractmethod
    def save_model(self, filename):
        pass        
        
    def task_model(self,task=-1):
        if self.singleheaded:
            return self.model
        else:
            return self.models[task]
    
    def task_fit(self, X, Y, task=None, validation_data=None, verbose=0):
        if not self.singleheaded:
            if task is None:
                raise Exception('Task number should be provided in task_fit if the model is not singleheaded')
            Y=Y[:,np.sum(Y,0)!=0]
            try:
                model = self.task_model(task)
            except:
                x = Dense(Y.shape[1],activation='softmax',name='output_task%d'%(len(self.models)+1))(self.model.output)
                task_Model = Model(self.model.input,x)
                task_Model.compile(loss=self.loss,optimizer=self.optimizer,metrics=self.metrics)
                self.models.append(task_Model)
                model = task_Model
        self.task_fit_method(X,Y,model,validation_data,verbose)
    
    def evaluate(self,X,Y,task=None,verbose=0):
        if not self.singleheaded:
            if task is None:
                raise Exception('Task number should be provided in evaluate if the model is not singleheaded')
            Y=Y[:,np.sum(Y,0)!=0]
            try:
                model = self.task_model(task)
            except:
                x = Dense(Y.shape[1],activation='softmax',name='output_task%d'%(len(self.models)+1))(self.model.output)
                task_Model = Model(self.model.input,x)
                task_Model.compile(loss=self.loss,optimizer=self.optimizer,metrics=self.metrics)
                self.models.append(task_Model)
                model = task_Model
        return self.task_model(task).evaluate(X,Y,verbose)
        
#    @abstractmethod
    def load_model(self, filename):
        pass