import mnist_data as mnist
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
import numpy as np
import time

mnist_all = mnist.get_MNIST_data()
mnist_train_rand, mnist_test_rand = mnist.MNIST_train_test_split_random(mnist_all)

mnist_train = list()
mnist_test = list()

for k in np.arange(50000,51000,1000):    
    mnist_train_k, mnist_test_k = mnist.MNIST_train_test_split_k(mnist_all, k)
    mnist_train.append(mnist_train_k)
    mnist_test.append(mnist_test_k)
    
    print mnist_train_k.targets.shape
    
    # Train using SVM
    t0 = time.clock()
    n_estimators = 50
    clf = OneVsRestClassifier(BaggingClassifier(SVC(C = 1.0, kernel = "rbf"), n_estimators = n_estimators, 
                                                max_samples=1.0/n_estimators, n_jobs = -1, verbose = True, bootstrap = False))
    clf.fit(mnist_train_k.data, mnist_train_k.targets)
    execTime = time.clock() - t0
    
    print execTime            
                         
    # Test using SVM
    y_pred = clf.predict(mnist_test_k.data)
    print metrics.accuracy_score(mnist_test_k.targets, y_pred)
    
    # Get scores
    
#    print mnist_train[(k-1)/1000].data.shape, mnist_train[(k-1)/1000].targets.shape
#    print mnist_test[(k-1)/1000].data.shape, mnist_test[(k-1)/1000].targets.shape
                     
mnist_train_fixed, mnist_test_fixed = mnist.MNIST_train_test_split(mnist_all)

print mnist_train_fixed.data.shape, mnist_test_fixed.targets.shape


