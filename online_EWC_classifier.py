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

class OnlineEWCClassifier(ContinualClassifier):
    def __init__(self, ewc_lambda=1, gamma=1, fisher_n=0, empirical=False, *args, **kwargs):
        self.ewc_lambda = ewc_lambda
        self.gamma=gamma
        self.mean = None
        self.precision = None
        self.task_count = 0
        self.fisher_n=fisher_n
        self.empirical=empirical        
        super().__init__(*args, **kwargs)
    
    def _save_model(self, objs):
        objs['ewc_lambda']=self.ewc_lambda
        objs['gamma']=self.gamma
        objs['mean']=self.mean
        objs['precision']=self.precision
        objs['task_count']=self.task_count
        objs['fisher_n']=self.fisher_n
        objs['empirical']=self.empirical
        
    
    def _load_model(self, objs):
        self.ewc_lambda=objs['ewc_lambda']
        self.gamma=objs['gamma']
        self.mean=objs['mean']
        self.precision=objs['precision']
        self.task_count=objs['task_count']
        self.fisher_n=objs['fisher_n']
        self.empirical=objs['empirical']
    
    def _task_fit(self, X, Y, model, new_task, batch_size, epochs, validation_data=None, verbose=2):
        self.inject_regularization(self.online_EWC)
        model.compile(loss=self.loss,optimizer=self.optimizer,metrics=['accuracy'])
        model.fit(X,Y, batch_size=batch_size, epochs=epochs, validation_data = validation_data, verbose=verbose, shuffle=True)
        if new_task:
            if self.empirical:
                self.update_laplace_approxiation_parameters(X,Y)
            else:
                self.update_laplace_approxiation_parameters(X)

    def update_laplace_approxiation_parameters(self,X,Y=None):
        model = self.task_model()
        len_weights = len(model.get_weights())-(not self.singleheaded)
        fisher_estimates = estimate_fisher_diagonal(model,X,Y,self.fisher_n,len_weights)
        self.mean = model.get_weights()
        if self.task_count>0:
            prev_prec = self.precision
            for i in range(0,len_weights):
                fisher_estimates[i]+=prev_prec[i]*self.gamma        
        self.precision = fisher_estimates
        self.task_count=1
        
            
    
    def online_EWC(self,weight_no):
        task_count = self.task_count
        if task_count is 0:
            def ewc_reg(weights):
                return 0
            return ewc_reg        
        if self.gamma is not 0:
            mean = self.mean[weight_no]
            prec = self.precision[weight_no]
            gamma = self.gamma
            def ewc_reg(weights):
                return self.ewc_lambda*0.5*K.sum((gamma*prec) * (weights-mean)**2)
            return ewc_reg
        return ewc_reg
        
        

