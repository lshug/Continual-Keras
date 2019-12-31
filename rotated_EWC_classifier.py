from continual_classifier import ContinualClassifier
from utils import estimate_fisher_diagonal
from keras.losses import categorical_crossentropy
import numpy as np
from keras.models import Model
from keras.layers import Dense, Activation
import tensorflow as tf
import keras.backend as K
from sklearn.utils import shuffle
from tqdm import tqdm

class RotatedEWCClassifier(ContinualClassifier):
    def __init__(self, ewc_lambda=1, fisher_n=0, rotate_n=0, empirical=False, *args, **kwargs):
        self.ewc_lambda = ewc_lambda
        self.means = []
        self.precisions = []
        self.task_count = 0
        self.fisher_n=fisher_n
        self.rotate_n=rotate_n
        self.empirical=empirical
        self.U1 = {}
        self.U2 = {}
        super().__init__(*args, **kwargs)
    
    def save_model_method(self, objs):
        objs['ewc_lambda']=self.ewc_lambda
        objs['means']=self.means
        objs['precisions']=self.precisions
        objs['task_count']=self.task_count
        objs['fisher_n']=self.fisher_n
        objs['rotate_n']=self.rotate_n
        objs['empirical']=self.empirical
        
    
    def load_model_method(self, objs):
        self.ewc_lambda=objs['ewc_lambda']
        self.means=objs['means']
        self.precisions=objs['precisions']
        self.task_count=objs['task_count']
        self.fisher_n=objs['fisher_n']
        self.rotate_n=objs['rotate_n']
        self.empirical=objs['empirical']
        self.U1={}
        self.U2={}
    
    def _task_fit(self, X, Y, model, new_task, batch_size, epochs, validation_data=None, verbose=2):
        if self.task_count == 0: #first task
            model.compile(loss=self.loss,optimizer=self.optimizer,metrics=['accuracy'])
            model.fit(X,Y, batch_size=batch_size, epochs=epochs, validation_data = validation_data, verbose=verbose, shuffle=True)
            self.task_count+=1
        elif new_task: #non-first task
            transformed_model = self.transform(model)            
            if self.empirical:
                self.update_laplace_approxiation_parameters(transformed_model,X,Y)
            else:
                self.update_laplace_approxiation_parameters(transformed_model,X)
            self.inject_regularization(self.EWC,transformed_model)
            self.task_count+=1            
            transformed_model.compile(loss=self.loss,optimizer=self.optimizer,metrics=['accuracy'])
            transformed_model.fit(X,Y, batch_size=batch_size, epochs=epochs, validation_data = validation_data, verbose=verbose, shuffle=True)
            self.combine(model,transformed_model)
        else: #old task            
            transformed_model = self.transform(model)
            self.inject_regularization(self.EWC,transformed_model)
            transformed_model.compile(loss=self.loss,optimizer=self.optimizer,metrics=['accuracy'])
            transformed_model.fit(X,Y, batch_size=batch_size, epochs=epochs, validation_data = validation_data, verbose=verbose, shuffle=True)
            self.combine(model,transformed_model)
        self.rotate(model,X,Y,self.rotate_n)

    def rotate(self,model,X,Y,rotate_n=0):
        if rotate_n is 0 or rotate_n>X.shape[0]:
            rotate_n = X.shape[0]
        rotated_layers=[]
        layer_inputs=[]
        layer_outputs=[]
        for l in model.layers:
            if ('Dense' in repr(l) or 'Conv2D' in repr(l)) and 'softmax' not in repr(l.activation) and 'Softmax' not in repr(l.output):                
                rotated_layers.append(l)                                    
                layer_inputs.append(l.input)
                layer_outputs.append(l.output)
        input_sums  = [np.zeros([l.input.shape[-1]]*2) for l in rotated_layers]
        output_sums = [np.zeros([l.output.shape[-1]]*2) for l in rotated_layers]        
        X=X[0:rotate_n]
        Y=Y[0:rotate_n]
        X,Y = shuffle(X,Y)        
        label_tensor = tf.where(tf.equal(tf.reduce_max(model.output,1,keepdims=True),model.output),tf.constant(1,shape=(1,model.output.shape[1])),tf.constant(0,shape=(1,model.output.shape[1])))        
        grads_tensor = K.gradients(categorical_crossentropy(label_tensor,model.output),layer_outputs)
        gradients = []
        sess=K.get_session()
        for i in tqdm(range(rotate_n), desc='Rotating'):
            gradient = sess.run(grads_tensor, feed_dict={model.input:np.array([X[i]])})
            for j in range(len(layer_outputs)):
                if 'Dense' in repr(rotated_layers[j]):
                    output_sums[j]+=np.dot(gradient[j].transpose(),gradient[j])/rotate_n
            inputs = sess.run(layer_inputs,feed_dict={model.input:np.array([X[i]])})
            for j in range(len(layer_inputs)):
                if 'Dense' in repr(rotated_layers[j]):
                    input_sums[j]+=np.dot(inputs[j].transpose(),inputs[j])/rotate_n
        for i in tqdm(range(len(output_sums)),desc='Doing SVDs'):
            if 'Dense' in repr(rotated_layers[j]):
                self.U1[rotated_layers[i]]=np.linalg.svd(input_sums[i], full_matrices=False)[0]
                self.U2[rotated_layers[i]]=np.linalg.svd(output_sums[i], full_matrices=False)[0]
                
    
    #order of U1, W, and U2 is U1.T@W@U2.T instead of U2.T@W@U1.T because keras dense implements a FC as y=xW+b
    def _replace_dense(self,dense_layer,input_tensor):
        U1 = self.U1[dense_layer]
        U2 = self.U2[dense_layer]
        name = dense_layer.name
        u1l = Dense(int(input_tensor.shape[-1]),name=name+'_U1')
        x=u1l(input_tensor)        
        u1w,u1b = u1l.get_weights()
        u1w = U1
        u1b = np.zeros_like(u1b)
        u1l.set_weights([u1w,u1b])
        u1l.trainable = False
        
                
        transformed_l = Dense(int(dense_layer.output.shape[-1]),name=name+'_transformed')
        x=transformed_l(x)
        W,b = dense_layer.get_weights()
        W = U1.transpose() @ W @ U2.transpose()
        transformed_l.set_weights([W,b])
        
        u2l = Dense(int(dense_layer.output.shape[-1]),name=name+'_U2')
        x=u2l(x)
        u2w,u2b = u2l.get_weights()
        u2w = U2
        u2b = np.zeros_like(u2b)
        u2l.set_weights([u2w,u2b])
        u2l.trainable = False
        x = Activation(dense_layer.activation)(x)
        return x
        
        
        
    #non-sequential layer replacement code based on https://stackoverflow.com/a/54517478
    def transform(self,model):
        #if not dense or conv2d then just add the layer
        #if dense/conv2d: replace it with 3 layers (named appropriately), followed by Activation
        network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}
        for layer in model.layers:
            for node in layer._outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name not in network_dict['input_layers_of']:
                    network_dict['input_layers_of'].update({layer_name: [layer.name]})
                else:
                    network_dict['input_layers_of'][layer_name].append(layer.name)
        network_dict['new_output_tensor_of'].update({model.layers[0].name: model.input})
        for layer in model.layers[1:]:
            layer_input = [network_dict['new_output_tensor_of'][layer_aux] for layer_aux in network_dict['input_layers_of'][layer.name]]
            if len(layer_input) == 1:
                layer_input = layer_input[0]
            if layer in self.U1.keys():
                if 'Dense' in repr(layer):
                    x = self._replace_dense(layer, layer_input)
                else:
                    pass
            else:
                x = layer(layer_input)
            network_dict['new_output_tensor_of'].update({layer.name: x})
        return Model(inputs=model.input,outputs=x)
        
    def combine(self,model,transformed_model):
        for l in model.layers:
            if l in self.U1.keys():
                U1 = self.U1[l]
                U2 = self.U2[l]
                transformed_W = transformed_model.get_layer(l.name+'_transformed').get_weights()[0]
                b = l.get_weights()[1]
                W = U1 @ transformed_W @ U2
                l.set_weights([W,b])
    
    def update_laplace_approxiation_parameters(self,model,X,Y=None):
        len_weights = len(K.batch_get_value(model.trainable_weights))-(not self.singleheaded)
        fisher_estimates = estimate_fisher_diagonal(model,X,Y,self.fisher_n,len_weights)        
        self.means.append(K.batch_get_value(model.trainable_weights))        
        self.precisions.append(fisher_estimates)

                        
    def EWC(self,weight_no):
        task_count = self.task_count
        if task_count is 0:
            def ewc_reg(weights):
                return 0
            return ewc_reg
        def ewc_reg(weights):
            loss_total = 0
            for i in range(1):
                loss_total+=self.ewc_lambda*0.5*K.sum((self.precisions[-1][weight_no]) * (weights-self.means[-1][weight_no])**2)
            return loss_total
        return ewc_reg
        
        


        
