import numpy as np
import pandas as pd
from keras.utils import to_categorical
from utils import rate_matrix
from EWC_classifier import EWCClassifier
'''
Tester should accept a ContinualClassifier and a list of tasks and output the following:
    -ACC
    -BWT
    -FWT
'''
    
class ContinualClassifierEvaluator():
    def __init__(self, classifier, tasks, labels, test_tasks=None, test_labels=None,task_order=None):
        self.classifier=classifier
        self.tasks=tasks
        self.labels=labels
        self.accuracies = np.zeros((len(tasks),len(tasks)))
        self.test_available = False
        self.task_order = task_order
        if test_tasks is not None:
            self.test_available = True
            if len(test_tasks) is not len(tasks):
                raise Exception('Training and testing task numbers do no match')
            self.test_tasks = test_tasks
            self.test_labels = test_labels
            self.test_accuracies = np.zeros((len(tasks),len(tasks)))
    
    def train(self,epochs=200,verbose=2):
        task_indices = range(len(self.tasks)) if self.task_order is None else self.task_order
        for i in task_indices:
            print('Training on task %d'%i)
            self.classifier.task_fit(self.tasks[i],self.labels[i],i,epochs=epochs,verbose=verbose)
            for j in range(len(self.tasks)):
                try:
                    self.accuracies[i,j] = self.classifier.evaluate(self.tasks[j],self.labels[j],j)[1]
                    if self.test_available:
                        self.test_accuracies[i,j] = self.classifier.evaluate(self.test_tasks[j],self.test_labels[j],j)[1]
                except:
                    pass
    def evaluate(self,on_test=False,save_accuracies_to_file=None):
        tasks = self.tasks
        labels = self.labels
        accuracies = self.accuracies
        if on_test:
            if not self.test_available:
                raise Exception('Testing task set not provided')
            tasks = self.test_tasks
            labels = self.test_labels
            accuracies = self.test_accuracies
        
        AAC = np.sum(accuracies[-1])/len(tasks)
        
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
            FWT+=accuracies[i-1,i] - self.classifier.evaluate(tasks[i],labels[i],i)[1]
        FWT = FWT/(len(tasks)-1)    
        self.classifier.model.set_weights(trained_weights)
        
        slope_metric = rate_matrix(accuracies)
        
        print('Metrics on {} set:'.format('test' if on_test else 'training'))
        print('AAC: {}\nBWT: {}\nFWT: {}\nAngle metric: {}'.format(AAC,BWT,FWT,slope_metric))
        if save_accuracies_to_file is not None:
            np.save(save_accuracies_to_file,self.accuracies)
            
        return AAC, BWT, FWT, slope_metric

