from collections import namedtuple
from sklearn.datasets import fetch_mldata
import numpy as np

MNIST_data = namedtuple("MNIST_data", "data, targets")              

def get_MNIST_data():
    data_dir = "C:\\Users\\vinot\\Desktop"
    mnist = fetch_mldata('MNIST original', data_home = data_dir)
    return mnist

def MNIST_train_test_split_random(mnist):                   
    train_indices = np.random.randint(0,70000,size=60000)
    mnist_train = MNIST_data(mnist.data[train_indices,:], mnist.target[train_indices])
    
#    test_indices = [i for i in np.arange(0,70000) if i not in train_indices]
    test_indices = np.setdiff1d(np.arange(0,70000), train_indices)
    mnist_test = MNIST_data(mnist.data[test_indices], mnist.target[test_indices])
    return mnist_train, mnist_test

def MNIST_train_test_split(mnist):                   
    mnist_train = MNIST_data(mnist.data[0:60000,:], mnist.target[0:60000])
    mnist_test = MNIST_data(mnist.data[60000:,:], mnist.target[60000:])
    
    #mnist_train = {'data': mnist.data[1:60001,:], 'target': mnist.target[1:60001]}
    #mnist_test = {'data': mnist.data[60000:,:], 'target': mnist.target[60000:]}    
    return mnist_train, mnist_test

def MNIST_train_test_split_k(mnist, k):
    train_indices = np.array([])
    sample_per_class = k/10

    for i in range(10):
        current_label = np.argwhere(mnist.target == i)
        train_indices = np.append(train_indices,current_label[:sample_per_class])
        train_indices = train_indices.astype(int)

    mnist_train = MNIST_data(mnist.data[train_indices,:], mnist.target[train_indices])
    
#    for i in xrange(10):
#        print mnist_train.targets[(i*sample_per_class):(i*sample_per_class + 10)]
    test_indices = np.arange(60000,70000)
    mnist_test = MNIST_data(mnist.data[test_indices], mnist.target[test_indices])
    
    return mnist_train, mnist_test