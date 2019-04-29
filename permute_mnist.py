import tensorflow as tf
import numpy as np
from tqdm import tqdm
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train=x_train.reshape(60000,784)/255.0
y_train=to_categorical(y_train)
def get_permute_mnist_tasks(task_no):
    tasks = []
    tasks.append(x_train)
    labels = []
    labels.append(y_train)
    for i in tqdm(range(task_no-1)):
        tasks.append(np.apply_along_axis(np.random.permutation,1,x_train))
        labels.append(y_train)
    return tasks, labels

