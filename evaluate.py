import numpy as np
import pandas as pd
from keras.utils import to_categorical
from EWC_classifier import EWCClassifier
'''
Tester should accept a ContinualClassifier and a list of tasks and output the following:
    -ACC
    -BWT
    -FWT
'''

def divide_dataset_into_tasks(X,Y,T):
    Y_categorical = to_categorical(Y)
    dropped = Y_categorical.shape[1] % T
    classes = Y_categorical.shape[1] - dropped
    if dropped is not 0:
        Y_categorical = Y_categorical[:,0:-1*dropped]
    per_task = classes/T
    tasks = []
    labels = []
    i=0
    while i<int(classes):
        X_t = None
        Y_t = None
        mask = None
        for j in range(0,int(per_task)):
            mask = (Y == i+j)
            if X_t is None:
                X_t = X[mask]
                Y_t = Y_categorical[mask]
            else:
                X_t=np.concatenate([X_t, X[mask]])
                Y_t=np.concatenate([Y_t, Y_categorical[mask]])
        
        tasks.append(X_t)
        labels.append(Y_t)
        i+=per_task
    return tasks, labels


class ContinualClassifierEvaluator():
    def __init__(self, classifier, tasks, labels):
        self.classifier=classifier
        self.tasks=tasks
        self.labels=labels
        self.accuracies = np.zeros((len(tasks),len(tasks)))
    
    def train(self,verbose=2):
        for i in range(len(self.tasks)):
            print('Training on task %d'%i)
            self.classifier.task_fit(self.tasks[i],self.labels[i],verbose=verbose)
            for j in range(len(self.tasks)):
                self.accuracies[i,j] = self.classifier.task_model(i).evaluate(tasks[j],labels[j])[1]
    
    def evaluate(self):
        ACC = np.sum(self.accuracies[-1])/len(self.tasks)
        
        BWT = 0
        for i in range(len(self.tasks)-1):
            BWT+=(self.accuracies[-1][i] - self.accuracies[i][i])
        BWT = BWT/(len(self.tasks)-1)
        
        trained_weights = self.classifier.model.get_weights()
        random_weights = []
        for i in range(len(trained_weights)):
            random_weights.append(np.random.rand(*(trained_weights[i].shape)))
        self.classifier.model.set_weights(random_weights)
        
        FWT = 0
        for i in range(1,len(self.tasks)):
            FWT+=self.accuracies[i,i] - self.classifier.task_model(i).evaluate(tasks[i],labels[i])[1]
        FWT = FWT/(len(self.tasks)-1)    
        self.classifier.model.set_weights(trained_weights)
        
        print('AAC: {} \n BWT: {} \n FWT: {}'.format(ACC,BWT,FWT))
        

