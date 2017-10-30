from collections import namedtuple
from sklearn.datasets import fetch_mldata
from sys import platform
from sklearn.decomposition import PCA
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

def MNIST_train_test_split_k(mnist, m, cv=False, k=0):
    """
    :param mnist: data.
    :param k: Number of training samples.
    :return: train, test data.
    """
    train_indices = np.array([])
    sample_per_class = m/10

    # Trying to get same number of samples per label
    for i in range(10): # class label
        current_label = np.argwhere(mnist.target[:60000] == i)
        
        # Checking if there is enough samples
        # Add at most @sample_per_class samples for each class in @train_indices.
        if np.size(current_label) < sample_per_class: # Not enough samples for the current label
            train_indices = np.append(train_indices,current_label)
        else:
            if not k:
                train_indices = np.append(train_indices,current_label[:sample_per_class])
            else:
                remain_cv_test = (np.size(current_label) - sample_per_class)
                if remain_cv_test <= 600:
                    remain_cv_train = np.setdiff1d(np.arange(np.size(current_label)),np.arange((k-1)*remain_cv_test,k*remain_cv_test))
                else:
                    remain_cv_train = np.setdiff1d(np.arange(np.size(current_label)),np.arange((k-1)*600,k*600))
                train_indices = np.append(train_indices,current_label[remain_cv_train])
    
    # Adding additional samples to get 'm' total samples
    # TODO: Do we need to do this? Adding additional samples will result
    # TODO: in more than @sample_per_class samples for some classes.
    # RVD: That's true. I had it here for borderline cases otherwise we'll never   
    # RVD: have all the 60000 samples if we call the function with say m = 60000 
    if np.size(train_indices) < m:
        remain_data = np.setdiff1d(np.arange(60000),train_indices)
        train_indices = np.append(train_indices,remain_data[:(m - np.size(train_indices))])
        
    train_indices = train_indices.astype(int)
    mnist_train = MNIST_data(mnist.data[train_indices,:], mnist.target[train_indices])
    
    if (not cv): # Check if cross-validation or actual train-test split
        test_indices = np.arange(60000,70000)
    else:
        test_indices = np.setdiff1d(np.arange(60000),train_indices)
        
    mnist_test = MNIST_data(mnist.data[test_indices], mnist.target[test_indices])
    
    return mnist_train, mnist_test

def MNIST_pca(mnist):
    variance = 0
    n_components = 87 # Changed from '5' for a faster run
 
    variance_arr = np.array([])
    n_components_arr = np.array([])
    
    # Mean centering
    mnist_data_centered = mnist.data - mnist.data.mean(axis=0)
    
    while (variance < 0.9):
        pca = PCA(n_components=n_components)
        pca.fit(mnist_data_centered[:60000,:])
        variance = np.sum(pca.explained_variance_ratio_)
        variance_arr = np.append(variance_arr,variance)
        n_components_arr = np.append(n_components_arr,n_components)
        n_components += 1
#        print n_components, variance
    
    plt.figure()
    plt.plot(n_components_arr,variance_arr)
    plt.xlabel('Number of components')
    plt.ylabel('Variance captured')
    plt.title('Variance vs. No. of components', fontweight='bold')
    plt.grid(linestyle='--')
#    plt.savefig('pca_variance_vs_no_comp.png',dpi = 600)
    plt.show()
    train_pca = pca.fit_transform(mnist_data_centered[:60000,:])
    
    return train_pca, mnist.data.mean(axis=0), pca.components_.T
    