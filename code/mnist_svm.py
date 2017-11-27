import mnist_data as mnist
import mnist_utils
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
def svm_linear_samples(C):
    training_data_percentage = np.array([1,10,20,30,40,50,60,70,80,90,100]) * 0.01
    stratified_rand_sample = mnist_utils.MNIST_Random_Sample_Stratified()
    length = len(training_data_percentage)
    svm_training_time = np.zeros((length,10), dtype=float)
    svm_prediction_time = np.zeros((length,10), dtype=float)
    svm_accuracy = np.zeros((length,10), dtype=float)

    for k in xrange(length):
        for i in range(10):
            
            mnist_train_k, mnist_test_k = stratified_rand_sample.sample_train_data(training_data_percentage[k])
            
            print "Number of training samples: {}".format(len(mnist_train_k.data))
            print "Number of test samples:{}".format(len(mnist_test_k.data))
            
            # One hidden layer with 100 neurons.
            t0 = time.clock()
            clf = OneVsOneClassifier(LinearSVC(C = C))
        #    clf = LinearSVC() Slower because it uses OneVsRest by default
            clf.fit(mnist_train_k.data, mnist_train_k.target)
            execTime = time.clock() - t0
            svm_training_time[k][i] = execTime
                        
            print "Execution Time for SVM: {}".format(execTime)                         
            
            t0 = time.clock()          
            y_pred = clf.predict(mnist_test_k.data)
            execTime = time.clock() - t0
            svm_prediction_time[k][i] = execTime
            
            accuracy_i = metrics.accuracy_score(mnist_test_k.target, y_pred)
            svm_accuracy[k][i] = accuracy_i
            
            print "Accuracy: {}".format(accuracy_i)
    
    np.savetxt("../experiments/expSVM/svm_training_times_different_tss.csv", svm_training_time, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/expSVM/svm_prediction_times_different_tss.csv", svm_prediction_time, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/expSVM/svm_accuracy_different_tss.csv", svm_accuracy, delimiter=",", fmt="%.3f")
            
    plt.figure()
    plt.grid(linestyle='--')
    plt.plot(training_data_percentage * 100, np.mean(svm_accuracy,axis=1),'b')
    plt.title('Accuracy vs. number of samples',fontsize=14, fontweight='bold')
    plt.xlabel('Percentage of samples')
    plt.ylabel('Accuracy')
    plt.savefig('..\Results_Plots\accuracy_vs_samples_svm_tuned.png', dpi = 600)
    plt.show()
            
# RBF kernel
def svm_rbf_samples(C):
    accuracy = np.array([])
    for k in xrange(60):    
        mnist_train_k, mnist_test_k = mnist.MNIST_train_test_split_k(mnist_all, (k+1)*1000)
        mnist_train.append(mnist_train_k)
        mnist_test.append(mnist_test_k)
        print "Number of training samples: {}".format((k+1)*1000)
        
        #Train using SVM (One v. One)
        t0 = time.clock()
        clf = OneVsOneClassifier(SVC(kernel = "rbf", C = C))
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
        fig_filename = ".\Results_Plots\SVM_RBF_ROC_samples_{}.png".format((k+1)*1000)
        fig_title = "ROC_RBF_SVM"
        res.plot_ROC_curve(mnist_test_k.target, y_score, fig_filename, fig_title)
            
    plt.figure()
    plt.grid(linestyle='--')
    plt.plot(np.arange(1000,61000,1000), accuracy,'b')
    plt.title('Accuracy vs. number of samples',fontsize=14, fontweight='bold')
    plt.xlabel('Number of samples')
    plt.ylabel('Accuracy')
#    plt.savefig('..\Results_Plots\accuracy_vs_samples_rbfsvm_default.png', dpi = 600)
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
                         

def svm_linear_param_C(C,train,test,k,use_pca=False):
    
    clf = LinearSVC(C = C)
    
    t0 = time.clock()
    clf.fit(train.data, train.target)
    execTime_train = time.clock() - t0
    
    t0 = time.clock()
    y_pred = clf.predict(test.data)
    execTime_predict = time.clock() - t0
    
    # Metrics
    accuracy = metrics.accuracy_score(test.target, y_pred)
    y_score = clf.decision_function(test.data)
    
    if use_pca:       
        precision = metrics.precision_score(test.target, y_pred, average='weighted')
        recall = metrics.recall_score(test.target, y_pred, average='weighted')
        cm = metrics.confusion_matrix(test.target, y_pred)
        classes = ['0','1','2','3','4','5','6','7','8','9']
        filename = '..\Results_Plots\SVM_Conf_matrix_C_{}_k_{}_using_pca.png'.format(C,k+1)
        res.confusion_matrix_plot(cm, classes, filename)
        return accuracy, y_score, execTime_train, execTime_predict, precision, recall
    
    else:
        return accuracy, y_score, execTime_train, execTime_predict
    
def cross_validation_k_fold(C, use_pca = False):
    accuracy = np.zeros(10)
    precision = np.zeros(10)
    recall = np.zeros(10)
    train_time = np.zeros(10)
    predict_time = np.zeros(10)

    if not use_pca:
        mnist_cv = mnist_all
    else:
        mnist_cv = mnist.MNIST_data(mnist_pca,mnist_all.target[:])
        
    cv_obj = mnist_utils.MNIST_CV_Stratified(mnist_cv)
    
    for k in xrange(10):
        
#        mnist_cv = mnist.MNIST_data(mnist_pca, mnist_all.target[:60000])
#        mnist_cv = mnist.MNIST_data(mnist_all.data[:60000,:], mnist_all.target[:60000])
#        train,test = mnist.MNIST_train_test_split_k(mnist_cv,54000,True,k+1)
        train, test = cv_obj.get_train_test_split(10, k)
        
        if use_pca:
            accuracy[k], y_score, train_time[k], predict_time[k], precision[k], recall[k] = svm_linear_param_C(C,train,test,k,True)
        else:
            accuracy[k], y_score, train_time[k], predict_time[k] = svm_linear_param_C(C,train,test,k)
        
        print "Accuracy = {} for k = {} and C = {}".format(accuracy[k],k+1,C)
        
        if not use_pca:
            filename = '..\Results_Plots\SVM_linear_ROC_C_{}_k_{}.png'.format(C,k+1)
        else:
            filename = '..\Results_Plots\SVM_linear_ROC_C_{}_k_{}_using_pca.png'.format(C,k+1)
            
        title = 'ROC_Linear_SVM'
        res.plot_ROC_curve(test.target,y_score,filename,title, True)
    
    if use_pca:
        np.savetxt("../experiments/expSVM/svm_precision_hypo_test.csv", precision, delimiter=",", fmt="%.3f")
        np.savetxt("../experiments/expSVM/svm_recall_hypo_test.csv", recall, delimiter=",", fmt="%.3f")
    
    return accuracy, train_time, predict_time