import tensorflow as tf
import numpy as np
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()



def permute(task_no):
    tasks = []
    tasks.append(x_train)
    labels = []
    labels.append(y_train)
    for i in range(task_no-1):
        tasks.append(np.apply_along_axis(np.random.permutation,1,x_train))
        labels.append(y_train)
    return tasks, labels

