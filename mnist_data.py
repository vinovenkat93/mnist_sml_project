from collections import namedtuple
from sklearn.datasets import fetch_mldata
import numpy as np
from sys import platform

MNIST_data = namedtuple("MNIST_data", "data, target")              

def get_MNIST_data():
    if platform == "darwin": # Mac OS
        data_dir = "../MNIST_Data"
    elif platform == "win32": # Windows
        data_dir = "D:\\Vinoth\\Course_work\\CS 578\\Project\\MNIST_Data"

    mnist = fetch_mldata('MNIST original', data_home = data_dir)
    return mnist

def MNIST_train_test_split(mnist):                   
    mnist_train = MNIST_data(mnist.data[:60000,:], mnist.target[:60000])
    mnist_test = MNIST_data(mnist.data[60000:,:], mnist.target[60000:])
      
    return mnist_train, mnist_test

def MNIST_train_test_split_k(mnist, k):
    """
    :param mnist: data.
    :param k: Number of training samples.
    :return: train, test data.
    """
    train_indices = np.array([])
    sample_per_class = k/10 # Number of samples per class.

    # Trying to get same number of samples per label
    for i in range(10): # class label
        current_label = np.argwhere(mnist.target[:60000] == i)
        
        # Checking if there is enough samples
        # Add at most @sample_per_class samples for each class in @train_indices.
        if np.size(current_label) < sample_per_class: # Not enough samples for the current label
            train_indices = np.append(train_indices,current_label)
        else:    
            train_indices = np.append(train_indices,current_label[:sample_per_class])
    
    # Adding additional samples to get 'k' total samples
    # TODO: Do we need to do this? Adding additional samples will result
    # TODO: in more than @sample_per_class samples for some classes.
    if np.size(train_indices) < k:
        remain_data = np.setdiff1d(np.arange(60000),train_indices)
        train_indices = np.append(train_indices,remain_data[:(k - np.size(train_indices))])
        
    train_indices = train_indices.astype(int)
    mnist_train = MNIST_data(mnist.data[train_indices,:], mnist.target[train_indices])
    
    # Test data stays the same
    test_indices = np.arange(60000,70000)
    mnist_test = MNIST_data(mnist.data[test_indices], mnist.target[test_indices])
    
    return mnist_train, mnist_test