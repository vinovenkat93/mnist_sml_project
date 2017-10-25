import mnist_data as mnist
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
import time

mnist_all = mnist.get_MNIST_data()

mnist_train = list()
mnist_test = list()

for k in xrange(60):    
    mnist_train_k, mnist_test_k = mnist.MNIST_train_test_split_k(mnist_all, (k+1)*1000)
    mnist_train.append(mnist_train_k)
    mnist_test.append(mnist_test_k)

    print "Number of training samples: {}".format((k+1)*1000)

    #Train using SVM (One v. One)
    t0 = time.clock()
    clf = OneVsOneClassifier(SVC(kernel = "rbf"))
    clf.fit(mnist_train_k.data, mnist_train_k.target)
    execTime = time.clock() - t0; 
    
    print "Execution Time for SVM (RBF Kernel: One vs. One): {}".format(execTime)                             
    y_pred = clf.predict(mnist_test_k.data)
    print "Accuracy: " + metrics.accuracy_score(mnist_test_k.target, y_pred)

for k in xrange(60):    
    mnist_train_k, mnist_test_k = mnist_train[k], mnist_test[k]
    
    #Train using SVM (One v. Rest)
    t0 = time.clock()
    clf = OneVsRestClassifier(SVC(kernel = "rbf"), n_jobs = -1)
    clf.fit(mnist_train_k.data, mnist_train_k.target)
    execTime = time.clock() - t0; 
    
    print "Execution Time for SVM (RBF Kernel: One vs. Rest): {}".format(execTime)                             
    y_pred = clf.predict(mnist_test_k.data)
    print "Accuracy: " + metrics.accuracy_score(mnist_test_k.target, y_pred)
    
for k in xrange(60):    
    mnist_train_k, mnist_test_k = mnist_train[k], mnist_test[k]
    
    # Train using SVM (Ensembles-faster)
    t0 = time.clock()
    n_estimators = 10
    clf = OneVsRestClassifier(BaggingClassifier(SVC(C = 1.0, kernel = "rbf"), n_estimators = n_estimators, 
                                                max_samples=1.0/n_estimators, n_jobs = -1, verbose = True, bootstrap = False))
    clf.fit(mnist_train_k.data, mnist_train_k.target)
    execTime = time.clock() - t0
    
    print "Execution Time for Ensemble SVM (RBF Kernel) with {} estimators: {}".format(n_estimators,execTime)
    y_pred = clf.predict(mnist_test_k.data)
    print "Accuracy: {}".format(metrics.accuracy_score(mnist_test_k.target, y_pred))
    
#    print mnist_train[(k-1)/1000].data.shape, mnist_train[(k-1)/1000].target.shape
#    print mnist_test[(k-1)/1000].data.shape, mnist_test[(k-1)/1000].target.shape
                     


