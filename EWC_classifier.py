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

class EWCClassifier(ContinualClassifier):
    def __init__(self, ewc_lambda=1, fisher_n=0, empirical=False, true_lambda=True, *args, **kwargs):
        self.ewc_lambda = ewc_lambda
        self.means = []
        self.precisions = []
        self.task_count = 0
        self.fisher_n=fisher_n
        self.empirical=empirical
        self.true_lambda = true_lambda
        super().__init__(*args, **kwargs)
    
    def save_model_method(self, objs):
        objs['ewc_lambda']=self.ewc_lambda
        objs['means']=self.means
        objs['precisions']=self.precisions
        objs['task_count']=self.task_count
        objs['fisher_n']=self.fisher_n
        objs['empirical']=self.empirical
        objs['true_lambda']=self.true_lambda
    
    def load_model_method(self, objs):
        self.ewc_lambda=objs['ewc_lambda']
        self.means=objs['means']
        self.precisions=objs['precisions']
        self.task_count=objs['task_count']
        self.fisher_n=objs['fisher_n']
        self.empirical=objs['empirical']
        self.true_lambda=objs['true_lambda']
    
    def _task_fit(self, X, Y, model, new_task, batch_size, epochs, validation_data=None, verbose=2):        
        model.compile(loss=self.loss,optimizer=self.optimizer,metrics=['accuracy'])
        model.fit(X,Y, batch_size=batch_size, epochs=epochs, validation_data = validation_data, verbose=verbose, shuffle=True)
        if new_task:
            if self.empirical:
                self.update_laplace_approxiation_parameters(X,Y)
            else:
                self.update_laplace_approxiation_parameters(X)
        self.inject_regularization(self.EWC)
    
    def update_laplace_approxiation_parameters(self,X,Y=None):
        model = self.task_model()
        len_weights = len(K.batch_get_value(model.trainable_weights))-2*(not self.singleheaded)
        fisher_estimates = estimate_fisher_diagonal(model,X,Y,self.fisher_n,len_weights,False,self.true_lambda)
        self.means.append(K.batch_get_value(model.trainable_weights))        
        self.precisions.append(fisher_estimates)
        self.task_count+=1
                        
    def EWC(self,weight_no):
        task_count = self.task_count
        if task_count is 0:
            def ewc_reg(weights):
                return 0
            return ewc_reg
        def ewc_reg(weights):
            loss_total = 0
            for i in range(task_count):
                    loss_total+=self.ewc_lambda*0.5*K.sum((self.precisions[i][weight_no]) * (weights-self.means[i][weight_no])**2)
            return loss_total
        return ewc_reg
        
        

