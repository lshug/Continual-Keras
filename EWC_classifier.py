from continual_classifier import ContinualClassifier
from keras.models import Model
import tensorflow as tf
import keras.backend as K
from sklearn.utils import shuffle


class EWCClassifier(ContinualClassifier):
    def __init__(self, shape, optimizer='adam', lr=0.00001, epochs=2, loss='categorical_crossentropy', metrics=['accuracy'], singleheaded_classes=None, model={'layers':3, 'units':400,'dropout':0,'activation':'relu'}, ewc_lambda=500, fisher_n=0, empirical=False, gamma=0):
        self.ewc_lambda = ewc_lambda
        self.modes = []
        self.precisions = []
        self.task_count = 0
        self.fisher_n=fisher_n
        self.empirical=empirical
        self.gamma=gamma
        super().__init__(shape,optimizer,lr,epochs,loss,metrics,singleheaded_classes,model)
    
    def save_model(self, filename):
        pass
        
    
    def load_model(self, filename):
        pass
    
    def task_fit_method(self, X, Y, validation_data=None, verbose=0):
        i = 0
        while True:
            try:
                l = self.model.get_layer(index=i)
                if len(l.trainable_weights)>0:
                    l.bias_regularizer=EWC(i)
                    i+=1
                    l.kernel_regularizer=EWC(i)
                    i+=1
            except:
                break
        
        if self.singleheaded:
            model = self.model
        else:
            model = self.models[-1]
        model.fit(X,Y,epochs = self.epochs, verbose=verbose, validation_data = validation_data, shuffle=True)
        self.estimate_fisher()
    
    
     
    def categorical_nll(y, logs):
        return -1*K.mean(tf.boolean_mask(logs,y))
    
    def estimate_fisher(self,X,Y):
        print('Fising\'')
        if self.singleheaded:
            model = self.model
        else:
            model = self.models[-1]
        len_weights = model.get_weights()
        if not self.singleheaded:
            len_weights-=1
        fisher_estimates = []
        for i in range(0,len_weights):
            fisher_estimates.append(np.zeros_like(model.get_weights()[i]))
        x = K.log(model.ouput)
        wrapped_model = Model(model.input,x)
        wrapped_model.compile(loss=categorical_nll,optimizer=self.optimizer,metrics=self.metrics)
        X,Y = shuffle(X,Y)
        fisher_n = self.fisher_n
        if self.fisher_n is 0:
            fisher_n = X.shape[0]
        for i in range(0,fisher_n):
            if self.empirical is False:
                label = wrapped_model.predict(np.array([X[i]]))[0]
            else:
                label = Y[i]
            gradients = K.get_session().run(K.gradients(wrapped_model.output,wrapped_model.trainable_weights), feed_dict={wrapped_model.input:np.array([X[i]])})
            for i in range(0,len_weights):
                fisher_estimates[i]+=gradients[i]**2
        for i in range(0,len_weights):
            fisher_estimates[i]=fisher_estimates[i]/fisher_n
        
        self.modes.append(model.get_weights())
        #Even if online, precision is stored separately after every task. Not very memory efficient, but this requires less coding desu :3
        if self.gamma is not None and self.task_count>0:
            prev_prec = self.precisions[-1]
            for i in range(0,len_weights):
                fisher_estimates[i]+=prev_prec[i]*self.gamma
        
        self.precisions.append(fisher_estimates)
        
        self.task_count+=1
        if self.gamma is not None:
            self.task_count=1
        
            
    
    def EWC(self,weight_no):
        '''returns a func that takes in weights,
        '''
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
            return self.EWC_lambda*0.5*K.sum((prec) * (weights-mean)**2)
        return ewc_reg

