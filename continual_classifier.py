import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
from abc import ABC, abstractmethod
import pickle

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
    layers). If singleheaded_classes is None, models are stored in self.models.
    """
    def __init__(self, optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'], singleheaded_classes=None,  model={'layers':3, 'units':400,'dropout':0,'activation':'relu'}):
        self.fitted_tasks = 0
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.regularizer_loaded = False
        if isinstance(model,dict):
            inp = Input(model['input_shape'],name='inputlayer')
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
            self.model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
        else:
            if singleheaded_classes is not None:
                x = Dense(singleheaded_classes,activation='softmax',name='singlehead')(x)
                self.singleheaded = True
            else:
                x = model.ouput
                self.singleheaded = False
                self.models = []
            self.model=Model(model.input,x)
            self.model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
            
    @abstractmethod
    def task_fit_method(self, X, Y, new_task, model, batch_size = 32, epochs=200, validation_data=None, verbose=0):
        pass
        
        

    def save_model(self, filename):
        objs = {}
        objs['fitted_tasks']=self.fitted_tasks
        objs['optimizer']=self.optimizer
        objs['loss']=self.loss
        objs['metrics'] = self.metrics
        objs['singleheaded']=self.singleheaded
        if self.singleheaded:
            objs['singlehead_config']=self.model.get_config()
            objs['singlehead_weights']=self.mode.get_weights()
        else:
            objs['base_config']=self.model.get_config()
            objs['base_weights']=self.model.get_weights()            
            objs['configs'] = [m.get_config() for m in models]
            objs['heads'] = [m.get_weights()[-1] for m in models]
        self.save_model_method(objs)
        pickle.dump(objs, open(filename,'wb'))

    @abstractmethod
    def save_model_method(self, objs):
        pass
    
    
    def load_model(self, filename):
        self.regularizer_loaded = False
        objs = pickle.load(open(filename,'rb')) 
        self.optimizer = objs['optimizer']
        self.loss=objs['loss']
        self.metrics=objs['metrics']
        self.singleheaded=objs['singleheaded']
        if self.singleheaded:
            self.model = Model.from_config(objs['singlehead_config'])
            self.model.set_weights(objs['singlehead_weights'])            
            self.model.compile(optimizer=self.optimizer,loss=self.loss,metrics=self.metrics)
        else:
            self.model = Model.from_config(objs['base_config'])
            base_weights = objs['base_weights']
            self.model.set_weights(base_weights)
            configs = objs['configs']
            heads  = objs['heads']
            models = []
            for i in range(len(configs)):
                model = Model.from_config(configs[i])
                body = base_weights[:]
                body.append(heads[i])
                model.set_weights(body)
                model.compile(optimizer=self.optimizer,loss=self.loss,metrics=self.metrics)
                models.append(model)
            self.models = models
        self.load_model_method(objs)
        

    @abstractmethod
    def load_model_method(self, objs):
        pass
    
    def task_model(self,task=-1):
        if self.singleheaded:
            return self.model
        else:   
            return self.models[task]
    
    def task_fit(self, X, Y, task=None, batch_size = 32, epochs=200, validation_data=None, verbose=0):
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
                
        model = self.task_model(task)
        self.task_fit_method(X,Y,model,new_task,batch_size=batch_size,epochs=epochs,validation_data=validation_data,verbose=verbose)
        if self.regularizer_loaded:
            self.clean_up_regularization()
            
    
    def evaluate(self,X,Y,task=None,batch_size=32,verbose=0):
        if not self.singleheaded:
            if task is None:
                raise Exception('Task number should be provided in evaluate if the model is not singleheaded')
            Y=Y[:,np.sum(Y,0)!=0]
            try:
                model = self.task_model(task)
            except:
                raise Exception('Could not retrive the head for task %d.'%task)
        return self.task_model(task).evaluate(X,Y,batch_size=batch_size,verbose=verbose)
    
    def predict(self,X,task=None,batch_size=32,verbose=0):
        if not self.singleheaded:
            if task is None:
                raise Exception('Task number should be provided in evaluate if the model is not singleheaded')
            Y=Y[:,np.sum(Y,0)!=0]
            try:
                model = self.task_model(task)
            except:
                raise Exception('Could not retrive the head for task %d.'%task)
        return self.task_model(task).predict(X,batch_size=batch_size,verbose=verbose)
    
    def inject_regularization(self,regularizer_generator):
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
            i+=1
            
    def clean_up_regularization(self):
        i = 0
        while True:            
            try:
                l = self.model.get_layer(index=i)
            except:
                break
            if len(l.trainable_weights)>0:
                l._losses=[]
            i+=1
        self.model.compile(loss=self.loss,optimizer=self.optimizer,metrics=self.metrics)
        

