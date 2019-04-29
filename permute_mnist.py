import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from keras.utils import to_categorical

def get_permute_mnist_tasks(task_no,samples_per_task=60000):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, y_train = shuffle(x_train,y_train)
    x_train=x_train.reshape(60000,784)/255.0
    y_train=to_categorical(y_train)
    x_train=x_train[0:samples_per_task]
    y_train=y_train[0:samples_per_task]
    tasks = []
    tasks.append(x_train)
    labels = []
    labels.append(y_train)
    for i in tqdm(range(task_no-1)):
        mask = np.random.permutation(784)
        tasks.append(np.apply_along_axis(lambda x: x[mask],1,x_train))
        labels.append(y_train)
    return tasks, labels

