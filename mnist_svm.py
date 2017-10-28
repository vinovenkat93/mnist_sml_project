import mnist_data as mnist
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
import numpy as np
import result_gen as res
import time

mnist_all = mnist.get_MNIST_data()
mnist_pca, _, _ = mnist.MNIST_pca(mnist_all)
mnist_train = list()
mnist_test = list()

# Linear Kernel
def svm_linear_samples():
    accuracy = np.array([])
    for k in xrange(60):    
        mnist_train_k, mnist_test_k = mnist.MNIST_train_test_split_k(mnist_all, (k+1)*1000)
        mnist_train.append(mnist_train_k)
        mnist_test.append(mnist_test_k)
        print "Number of training samples: {}".format((k+1)*1000)
        
        #Train using SVM (One v. One)
        t0 = time.clock()
        clf = OneVsOneClassifier(LinearSVC())
        #    clf = LinearSVC() Slower because it uses OneVsRest by default
        clf.fit(mnist_train_k.data, mnist_train_k.target)
        execTime = time.clock() - t0; 
        
        print "Execution Time for SVM (Linear Kernel: One vs. One): {}".format(execTime)                             
        y_pred = clf.predict(mnist_test_k.data)
        
        # Metrics
        accuracy_i = metrics.accuracy_score(mnist_test_k.target, y_pred)    
        accuracy = np.append(accuracy, accuracy_i)
        
        # ROC curves for each iteration for all classes
        y_score = clf.decision_function(mnist_test_k.data);
        
        # Plot ROC curves
        fig_filename = "SVM_linear_ROC_samples_{}.png".format((k+1)*1000)
        fig_title = "ROC_Linear_SVM"
        res.plot_ROC_curve(mnist_test_k.target, y_score, fig_filename, fig_title)
            
    plt.figure()
    plt.grid(linestyle='--')
    plt.plot(np.arange(1000,61000,1000), accuracy,'b')
    plt.title('Accuracy vs. number of samples',fontsize=14, fontweight='bold')
    plt.xlabel('Number of samples')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_vs_samples_linearsvm_default.png', dpi = 600)
    plt.show()
            
# RBF kernel
def svm_rbf_samples():
    accuracy = np.array([])
    for k in xrange(60):    
        mnist_train_k, mnist_test_k = mnist.MNIST_train_test_split_k(mnist_all, (k+1)*1000)
        mnist_train.append(mnist_train_k)
        mnist_test.append(mnist_test_k)
        print "Number of training samples: {}".format((k+1)*1000)
        
        #Train using SVM (One v. One)
        t0 = time.clock()
        clf = OneVsOneClassifier(SVC(kernel = "rbf"))
        #    clf = OneVsRestClassifier(SVC(kernel = "rbf")) Slower than OneVsOne (which scales better with n_samples)
        clf.fit(mnist_train_k.data, mnist_train_k.target)
        execTime = time.clock() - t0; 
        
        print "Execution Time for SVM (RBF Kernel: One vs. One): {}".format(execTime)                                                          
        y_pred = clf.predict(mnist_test_k.data)
        
        # Metrics
        accuracy_i = metrics.accuracy_score(mnist_test_k.target, y_pred)    
        accuracy = np.append(accuracy, accuracy_i)
        
        # ROC curves for each iteration for all classes
        y_score = clf.decision_function(mnist_test_k.data);
        
        # Plot ROC curves
        fig_filename = "SVM_RBF_ROC_samples_{}.png".format((k+1)*1000)
        fig_title = "ROC_RBF_SVM"
        res.plot_ROC_curve(mnist_test_k.target, y_score, fig_filename, fig_title)
            
    plt.figure()
    plt.grid(linestyle='--')
    plt.plot(np.arange(1000,61000,1000), accuracy,'b')
    plt.title('Accuracy vs. number of samples',fontsize=14, fontweight='bold')
    plt.xlabel('Number of samples')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_vs_samples_linearsvm_default.png', dpi = 600)
    plt.show()                        

# SVM Ensembles RBF
def svm_ensembles_rbf_samples():
    for k in xrange(60): 
        mnist_train_k, mnist_test_k = mnist.MNIST_train_test_split_k(mnist_all, (k+1)*1000)
        mnist_train.append(mnist_train_k)
        mnist_test.append(mnist_test_k)
        print "Number of training samples: {}".format((k+1)*1000)
        
        # Train using SVM (Ensembles-faster) (Still slower than Linear kernal, but have to check accuracy to confirm) 
        t0 = time.clock()
        n_estimators = 10
        clf = OneVsRestClassifier(BaggingClassifier(SVC(C = 1.0, kernel = "rbf"), n_estimators = n_estimators, 
                                                    max_samples=1.0/n_estimators, n_jobs = 1, verbose = True, bootstrap = False))
        clf.fit(mnist_train_k.data, mnist_train_k.target)
        execTime = time.clock() - t0
        
        print "Execution Time for Ensemble SVM (RBF Kernel) with {} estimators: {}".format(n_estimators,execTime)
        y_pred = clf.predict(mnist_test_k.data)
        print "Accuracy: {}".format(metrics.accuracy_score(mnist_test_k.target, y_pred))
                         

def svm_linear_param_C(C,train,test):
    
    clf = LinearSVC(C = C)
    clf.fit(train.data, train.target)
    
    y_pred = clf.predict(test.data)
    
    # Metrics
    accuracy = metrics.accuracy_score(test.target, y_pred)
    y_score = clf.decision_function(test.data);
    
    return accuracy, y_score
    
def cross_validation_k_fold(C):
    accuracy = np.zeros(10)
    for k in xrange(10):
        
        mnist_cv = mnist.MNIST_data(mnist_pca, mnist_all.target[:60000])
        train,test = mnist.MNIST_train_test_split_k(mnist_cv,54000,True,k+1)
        accuracy[k], y_score = svm_linear_param_C(C,train,test)
        
        print "Accuracy = {} for k = {} and C = {}".format(accuracy[k],k+1,C)
        
        filename = 'SVM_linear_ROC_C_{}_k_{}.png'.format(C,k+1)
        title = 'ROC_Linear_SVM'
        res.plot_ROC_curve(test.target,y_score,filename,title, True)
        
    return accuracy