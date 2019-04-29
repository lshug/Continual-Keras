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

def split_train_test(tasks,labels,fraction=0.2):
    new_tasks = []
    new_labels = []
    test_tasks = []
    test_labels = []
    for i in range(len(tasks)):
        last_n=int(tasks[i].shape[0]*fraction)
        new_tasks.append(tasks[i][0:last_n])
        new_labels.append(labels[i][0:last_n])
        test_tasks.append(tasks[i][last_n:])
        test_labels.append(labels[i][last_n:])        
    return new_tasks,new_labels,test_tasks,test_labels
    
class ContinualClassifierEvaluator():
    def __init__(self, classifier, tasks, labels, test_tasks=None, test_labels=None):
        self.classifier=classifier
        self.tasks=tasks
        self.labels=labels
        self.accuracies = np.zeros((len(tasks),len(tasks)))
        self.test_available = False
        if test_tasks is not None:
            self.test_available = False
            if len(test_tasks) is not len(tasks):
                raise Exception('Training and testing task numbers do no match')
            self.test_tasks = test_tasks
            self.test_labels = test_labels
            self.test_accuracies = np.zeros((len(tasks),len(tasks)))
    
    def train(self,verbose=2):
        for i in range(len(self.tasks)):
            print('Training on task %d'%i)
            self.classifier.task_fit(self.tasks[i],self.labels[i],i,verbose=verbose)
            for j in range(len(self.tasks)):
                self.accuracies[i,j] = self.classifier.evaluate(self.tasks[j],self.labels[j],j)[1]
                if self.test_available:
                    self.test_accuracies[i,j] = self.classifier.evaluate(self.test_tasks[j],self.test_labels[j],j)[1]
    
    def evaluate(self,on_test=False,save_accuracies_to_file=None,):
        tasks = self.tasks
        labels = self.labels
        accuracies = self.accuracies
        if on_test:
            if not self.test_available:
                raise Exception('Testing task set not provided')
            tasks = test_tasks
            labels = test_labels
            accuracies = self.test_accuracies
        
        ACC = np.sum(accuracies[-1])/len(tasks)
        
        BWT = 0
        for i in range(len(tasks)-1):
            BWT+=(accuracies[-1][i] - accuracies[i][i])
        BWT = BWT/(len(tasks)-1)
        
        trained_weights = self.classifier.model.get_weights()
        random_weights = []
        for i in range(len(trained_weights)):
            random_weights.append(np.random.rand(*(trained_weights[i].shape)))
        self.classifier.model.set_weights(random_weights)
        
        FWT = 0
        for i in range(1,len(tasks)):
            FWT+=accuracies[i,i] - self.classifier.evaluate(tasks[i],labels[i],i)[1]
        FWT = FWT/(len(tasks)-1)    
        self.classifier.model.set_weights(trained_weights)
        
        print('AAC: {} \n BWT: {} \n FWT: {}'.format(ACC,BWT,FWT))
        if save_accuracies_to_file is not None:
            np.save(save_accuracies_to_file,self.accuracies)
        

