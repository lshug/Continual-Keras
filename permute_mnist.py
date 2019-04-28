import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# generates array of size n, where array[0] holds 784 pictures from MNIST, array[:n] are random permutated pixel images
def permute(n):

  output = []; temp = []
  # put normal MNIST data into array
  data = x_train[:784]
  for d in data:
    temp.append(d)
  output.append(temp)
  
  # put permuted MNIST in the array
  for m in range(1, n):
    permutations = []
    for k in range(784):
      
      # random indexes for matrix
      x = np.random.permutation(28)
      y = np.random.permutation(28)

      img = [[0 for i in range(28)] for j in range(28)] # initialize empty 28x28 array

      # shuffle pixels in image
      for i in range(28):
        for j in range(28):
          _x = x[i]; _y = y[j]
          img[i][j] = x_train[k][_x][_y]
      permutations.append(img)
    output.append(permutations)

  return output