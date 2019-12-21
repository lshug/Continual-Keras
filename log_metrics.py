import numpy as np
from classification_evaluator import ContinualClassifierEvaluator
from utils import divide_dataset_into_tasks,split_train_test,get_permute_mnist_tasks,load_vowels
from EWC_classifier import EWCClassifier
import os

#extensive test

def do_evals(tasks, labels):
    tasks, labels, test_tasks, test_labels = split_train_test(tasks,labels)
    ewc = EWCClassifier((tasks[0].shape[1],),fisher_n=3000,epochs=5,batch=20,ewc_lambda=3,lr=0.1,optimizer='sgd',model={'layers':2, 'units':100,'dropout':0,'activation':'relu'})
    evaluator = ContinualClassifierEvaluator(ewc, tasks, labels, test_tasks, test_labels)
    evaluator.train(verbose=1)
    train_metrics = evaluator.evaluate()
    test_metrics = evaluator.evaluate(True)
    return train_metrics,test_metrics

#perm mnist
mnist_train_metrics = []
mnist_test_metrics  = []
for i in range(3,20):
    tasks, labels = get_permute_mnist_tasks(i,1250)
    train, test = do_evals(tasks, labels)
    mnist_train_metrics.append(train)
    mnist_test_metrics.append(test)
mnist_train_metrics = np.array(mnist_train_metrics)
mnist_test_metrics = np.array(mnist_test_metrics)
np.save('logs/mnist_train_metrics.py',mnist_train_metrics)
np.save('logs/mnist_test_metrics.py',mnist_test_metrics)


#omniglot
omniglot_train_metrics = []
omniglot_test_metrics  = []
X = np.load('char.npy')
Y = np.load('labels.npy')
tasks, labels = divide_dataset_into_tasks(X,Y,5)
for i in range(3,20):
    t = tasks[0:i]
    l = labels[0:i]
    train, test = do_evals(t, l)
    omniglot_train_metrics.append(train)
    omniglot_test_metrics.append(test)
omniglot_train_metrics = np.array(omniglot_train_metrics)
omniglot_test_metrics = np.array(omniglot_test_metrics)
np.save('logs/omniglot_train_metrics.py',omniglot_train_metrics)
np.save('logs/omniglot_test_metrics.py',omniglot_test_metrics)


#omniglot - accuracy matrices
tasks, labels, test_tasks, test_labels = split_train_test(tasks,labels)
ewc = EWCClassifier((tasks[0].shape[1],),fisher_n=3000,epochs=5,batch=20,ewc_lambda=3,lr=0.1,optimizer='sgd',model={'layers':2, 'units':100,'dropout':0,'activation':'relu'})
evaluator = ContinualClassifierEvaluator(ewc, tasks, labels, test_tasks, test_labels)
evaluator.train(verbose=1)
train_metrics = evaluator.evaluate(save_accuracies_to_file='logs/omniglot_train_accuracies.npy')
test_metrics = evaluator.evaluate(True,save_accuracies_to_file='logs/omniglot_train_accuracies.npy')


#perm mnist - accuracy matrices
tasks, labels = get_permute_mnist_tasks(20,1250)
tasks, labels, test_tasks, test_labels = split_train_test(tasks,labels)
ewc = EWCClassifier((tasks[0].shape[1],),fisher_n=3000,epochs=5,batch=20,ewc_lambda=3,lr=0.1,optimizer='sgd',model={'layers':2, 'units':100,'dropout':0,'activation':'relu'})
evaluator = ContinualClassifierEvaluator(ewc, tasks, labels, test_tasks, test_labels)
evaluator.train(verbose=1)
train_metrics = evaluator.evaluate(save_accuracies_to_file='logs/mnist_train_accuracies.npy')
test_metrics = evaluator.evaluate(True,save_accuracies_to_file='logs/mnist_train_accuracies.npy')

