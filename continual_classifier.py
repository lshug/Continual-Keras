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
        self.fitted_tasks = 0
        self.epochs = epochs
        self.batch=batch
        optim = Adam(lr)
        if optimizer is 'sgd':
            optim = SGD(lr)
        self.optimizer = optim
        self.loss = loss
        self.metrics = metrics
        self.regularizer_loaded = False
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
    def task_fit_method(self, X, Y, new_task, model, validation_data=None, verbose=0):
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
        new_task = False
        if task is 0 or task is None or task is self.fitted_tasks:
            self.fitted_tasks+=1
            new_task = True
        else:
           raise Exception('Task numbers must be sequential without any gaps (e.g. task 6 cannot follow immediately after task 4)')
        if not self.singleheaded:
            if task is None:
                task = self.fitted_tasks
            Y=Y[:,np.sum(Y,0)!=0]
            try:
                model = self.task_model(task)
            except:
                x = Dense(Y.shape[1],activation='softmax',name='output_task%d'%(len(self.models)+1))(self.model.output)
                task_Model = Model(self.model.input,x)
                task_Model.compile(loss=self.loss,optimizer=self.optimizer,metrics=self.metrics)
                self.models.append(task_Model)
                model = task_Model
        else:
            model = self.task_model(task)
        self.task_fit_method(X,Y,model,new_task,validation_data,verbose)
        if self.regularizer_loaded:
            self.clean_up_regularization()
            
    
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
        return self.task_model(task).evaluate(X,Y,batch_size=self.batch,verbose=verbose)
    
    def inject_regularization(regularizer_generator):
        self.regularizer_loaded = True
        i = 0
        j = 0
        while True:            
            try:
                l = self.model.get_layer(index=i)
            except:
                break
            for k in l.trainable_weights:
                l.add_loss(regularizer_generator(j)(k))
                j+=1
                l.add_loss(regularizer_generator(j)(k))
                j+=1                
            i+=1
            
    def clean_up_regularization():
        i = 0
        while True:            
            try:
                l = self.model.get_layer(index=i)
            except:
                break
            if len(l.trainable_weights)>0:
                l._losses=[]
            i+=1
        model.compile(loss=self.loss,optimizer=self.optimizer,metrics=['accuracy'])
        
#    @abstractmethod
    def load_model(self, filename):
        pass
