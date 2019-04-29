from continual_classifier import ContinualClassifier
import numpy as np
from keras.models import Model
from keras.layers import Lambda
import tensorflow as tf
import keras.backend as K
from sklearn.utils import shuffle
from tqdm import tqdm

def categorical_nll(y, x):
    return -1*K.mean(tf.boolean_mask(tf.log(x),y))

class EWCClassifier(ContinualClassifier):
    def __init__(self, shape, optimizer='adam',batch=32, lr=0.0005, epochs=150, metrics=['accuracy'], singleheaded_classes=None, model={'layers':3, 'units':200,'dropout':0,'activation':'relu'}, ewc_lambda=500, fisher_n=0, empirical=False, gamma=0):
        self.ewc_lambda = ewc_lambda
        self.modes = []
        self.precisions = []
        self.task_count = 0
        self.fisher_n=fisher_n
        self.empirical=empirical
        self.gamma=gamma
        super().__init__(shape,optimizer,batch,lr,epochs,categorical_nll,metrics,singleheaded_classes,model)
    
    def save_model(self, filename):
        pass
        
    
    def load_model(self, filename):
        pass
    
    def task_fit_method(self, X, Y, model, validation_data=None, verbose=2):
        i = 0
        j = 0
        while True:
            try:
                l = self.model.get_layer(index=i)
                if len(l.trainable_weights)>0:
                    l.bias_regularizer=EWC(j)
                    j+=1
                    l.kernel_regularizer=EWC(j)
                    j+=1
                i+=1
            except:
                break
        model.fit(X,Y,epochs = self.epochs, batch_size=self.batch, verbose=verbose, validation_data = validation_data, shuffle=True)
        self.estimate_fisher(X,Y)
    
    def estimate_fisher(self,X,Y):
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
        #x = Lambda(lambda l: K.log(l))(model.output)
        wrapped_model = Model(model.input,model.output)
        wrapped_model.compile(loss=categorical_nll,optimizer=self.optimizer,metrics=self.metrics)
        X,Y = shuffle(X,Y)
        fisher_n = self.fisher_n
        if self.fisher_n is 0 or self.fisher_n>X.shape[0]:
            fisher_n = X.shape[0]
        
        X=X[0:fisher_n]
        if self.empirical is False:            
            label=wrapped_model.predict(X)
        else:
            label=Y[0:fisher_n]
            
        #for each x,y pair, count the gradient with respect to the loss.
        gradients = []
        sess=K.get_session()
        y_placeholder = tf.placeholder(tf.float32, shape=Y[0].shape)
        grads_tesnor = K.gradients(categorical_nll(y_placeholder,wrapped_model.output),wrapped_model.trainable_weights)
        for i in tqdm(range(fisher_n)):
            gradients.append(sess.run(grads_tesnor, feed_dict={y_placeholder:Y[i],wrapped_model.input:np.array([X[i]])}))
        
        
        
        for i in tqdm(range(fisher_n)):
            for j in range(len_weights):
                fisher_estimates[j]+=gradients[i][j]**2
        
        for i in range(0,len_weights):
            fisher_estimates[i]=fisher_estimates[i]/fisher_n
        
        self.modes.append(model.get_weights())
        #Even if online, precision and mean are stored separately after every task. Not very memory efficient, but this requires less coding desu :3
        if self.gamma is not None and self.task_count>0:
            prev_prec = self.precisions[-1]
            for i in range(0,len_weights):
                fisher_estimates[i]+=prev_prec[i]*self.gamma
        
        self.precisions.append(fisher_estimates)
        
        self.task_count+=1
        if self.gamma is not None:
            self.task_count=1
        
            
    
    def EWC(self,weight_no):
        mean = self.means[-1][weight_no]
        prec = self.precisions[-1][weight_no]
        gamma = self.gamma
        task_count = self.task_count
        if task_count is 0:
            def ewc_reg(weights):
                return 0
            return ewc_reg
        if gamma is not 0:
            def ewc_reg(weights):
                return self.ewc_lambda*0.5*K.sum((gamma*prec) * (weights-mean)**2)
            return ewc_reg
        def ewc_reg(weights):
            loss_total = None
            for i in range(task_count):
                if loss_total is None:
                    loss_total=self.EWC_lambda*0.5*K.sum((prec[i][weight_no]) * (weights-self.means[i][weight_no])**2)
                else:
                    loss_total+=self.EWC_lambda*0.5*K.sum((prec[i][weight_no]) * (weights-self.means[i][weight_no])**2)
            return loss_total
        return ewc_reg
        
        

