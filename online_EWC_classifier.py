from continual_classifier import ContinualClassifier
from keras.losses import categorical_crossentropy
import numpy as np
from keras.models import Model
from keras.layers import Lambda
import tensorflow as tf
import keras.backend as K
from sklearn.utils import shuffle
from tqdm import tqdm

class OnlineEWCClassifier(ContinualClassifier):
    def __init__(self, ewc_lambda=1, fisher_n=0, empirical=False, gamma=1, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], singleheaded_classes=None, model={'layers':3, 'units':200,'dropout':0,'activation':'relu'}):
        self.ewc_lambda = ewc_lambda
        self.mean = None
        self.precision = None
        self.task_count = 0
        self.fisher_n=fisher_n
        self.empirical=empirical
        self.gamma=gamma
        super().__init__(optimizer,loss,metrics,singleheaded_classes,model)
    
    def save_model(self, filename):
        pass
        
    
    def load_model(self, filename):
        pass
    
    def task_fit_method(self, X, Y, model, new_task, batch_size, epochs, validation_data=None, verbose=2):
        i = 0
        j = 0
        if new_task:
            self.inject_regularization(self.EWC)
        model.compile(loss=self.loss,optimizer=self.optimizer,metrics=['accuracy'])
        model.fit(X,Y, batch_size=batch_size, epochs=epochs, validation_data = validation_data, verbose=verbose, shuffle=True)
        if new_task:
            self.estimate_fisher(X,Y)
            
    
    def estimate_fisher(self,X,Y=None):
        if self.singleheaded:
            model = self.model
        else:
            model = self.models[-1]
        len_weights = len(model.get_weights())
        if not self.singleheaded:
            len_weights-=1
        fisher_estimates = []
        for i in range(0,len_weights):
            fisher_estimates.append(np.zeros_like(model.get_weights()[i]))
        wrapped_model = Model(model.input,model.output)
        wrapped_model.compile(loss=self.loss,optimizer=self.optimizer,metrics=self.metrics)
        
        fisher_n = self.fisher_n
        if self.fisher_n is 0 or self.fisher_n>X.shape[0]:
            fisher_n = X.shape[0]
        
        X=X[0:fisher_n]
        if self.empirical is False:
            X = np.random.permutation(X)
            label=wrapped_model.predict(X)
            label = np.squeeze(np.eye(label.shape[1])[np.argmax(label,-1).reshape(-1)])
        else:
            X,Y = shuffle(X,Y)
            label=Y[0:fisher_n]
            
        #for each x,y pair, count the gradient with respect to the loss.
        gradients = []
        sess=K.get_session()
        y_placeholder = tf.placeholder(tf.float32, shape=label[0].shape)
        grads_tesnor = K.gradients(categorical_crossentropy(y_placeholder,wrapped_model.output),wrapped_model.trainable_weights)
        for i in tqdm(range(fisher_n)):
            gradients.append(sess.run(grads_tesnor, feed_dict={y_placeholder:label[i],wrapped_model.input:np.array([X[i]])}))
        
        
        
        for i in tqdm(range(fisher_n)):
            for j in range(len_weights):
                fisher_estimates[j]+=gradients[i][j]**2  #Since we're only going to use the diagonal of Fisher, rather than calculate the whole outer product we can get just the diagonal by squaring each element of the gradient. 
        
        for i in range(0,len_weights):
            fisher_estimates[i]=fisher_estimates[i]/fisher_n 
        
        self.mean = model.get_weights()
        
        #self.means.append(model.get_weights())
        #Even if online, precision and mean are stored separately after every task. Not very memory efficient, but this requires less coding desu :3
        if self.task_count>0:
            prev_prec = self.precision
            for i in range(0,len_weights):
                fisher_estimates[i]+=prev_prec[i]*self.gamma
        
        self.precision = fisher_estimates
        self.task_count=1
        
            
    
    def EWC(self,weight_no):
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
        
        

