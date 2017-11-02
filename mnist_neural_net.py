import mnist_data as mnist
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import result_gen as res
import time

mnist_all = mnist.get_MNIST_data()

mnist_train = list()
mnist_test = list()

"""
Studying the effect of samples
"""
def neural_net_samples():
    accuracy = np.array([])
    for k in xrange(60):  
        mnist_train_k, mnist_test_k = mnist.MNIST_train_test_split_k(mnist_all, (k + 1)*1000)
        mnist_train.append(mnist_train_k)
        mnist_test.append(mnist_test_k)
        
        print "Number of training samples: {}".format((k+1)*1000)
        
        # One hidden layer with 100 neurons.
        t0 = time.clock()
        mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=200, alpha=1e-4,
                            solver='sgd', tol=1e-4, random_state=1,
                            learning_rate_init=.001)
        mlp.fit(mnist_train_k.data, mnist_train_k.target)
        execTime = time.clock() - t0
                    
        print "Execution Time for Neural Net: {}".format(execTime)                         
        
        y_pred = mlp.predict(mnist_test_k.data)
        accuracy_i = metrics.accuracy_score(mnist_test_k.target, y_pred)
        accuracy = np.append(accuracy, accuracy_i)
        
        print "Accuracy: {}".format(accuracy_i)
        
    plt.figure()
    plt.grid(linestyle='--')
    plt.plot(np.arange(1000,61000,1000), accuracy,'b')
    plt.title('Accuracy vs. number of samples',fontsize=14, fontweight='bold')
    plt.xlabel('Number of samples')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_vs_samples_neural_net_default.png', dpi = 600)
    plt.show()

"""
Optimizing the size of hidden layers
"""
def neural_net_layers_size():
    
    mnist_train_lay, mnist_test_lay = mnist.MNIST_train_test_split(mnist_all)
    hidden_layer_sizes_list = [(50), (100), (100,100), (50,100), (100,50), (200,200)]
    
    for hidden_layer_sizes in hidden_layer_sizes_list:
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=400, alpha=1e-4,
                                solver='sgd', tol=1e-4, random_state=1,
                                learning_rate_init=.0011, learning_rate='adaptive')
        mlp.fit(mnist_train_lay.data, mnist_train_lay.target)                       
        
        accuracy = mlp.score(mnist_test_lay.data, mnist_test_lay.target)        
#        y_pred = mlp.predict(mnist_test_lay.data)
#        accuracy = metrics.accuracy_score(mnist_test_lay.target, y_pred)
    
        print "Accuracy: {} for Hidden Layer sizes: {}".format(accuracy, hidden_layer_sizes)

"""
Optimizing neural net params         
"""
def neural_net_opt(alpha, act_fcn, learn_rate, train, test):
    mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=200, alpha=alpha,
                            solver='sgd', tol=1e-4, random_state=1,
                            learning_rate_init=learn_rate, activation=act_fcn)
    mlp.fit(train.data, train.target)
    
    y_pred = mlp.predict(test.data)
    
    # Metrics
    accuracy = metrics.accuracy_score(test.target, y_pred)
    y_score = mlp.predict_proba(test.data);
    
    return accuracy, y_score
    
"""
10-fold CV for param optimization
"""
def cross_validation_k_fold(alpha = 'default', act_fcn = 'default', learn_rate = 'default'):
    """
    :param alpha: Regularization term
    :param act_fcn: Activation function for the net
    :param learn_rate: Learning rate for the net
    :return: Accuracy score for each fold in the 10-fold CV
    """
    a = 0; l = 0; ac = 0
    accuracy = np.zeros(10)
    for k in xrange(10):
        
        #        mnist_cv = mnist.MNIST_data(mnist_pca, mnist_all.target[:60000])
        mnist_cv = mnist.MNIST_data(mnist_all.data[:60000,:], mnist_all.target[:60000])

        train,test = mnist.MNIST_train_test_split_k(mnist_cv,54000,True,k+1)
        if alpha == 'default': alpha = 1e-4; a = 1
        if learn_rate == 'default': learn_rate = 1e-3; l = 1
        if act_fcn == 'default': act_fcn = 'relu'; ac = 1
        
        accuracy[k], y_score = neural_net_opt(alpha, act_fcn, learn_rate, train, test)
        
        # Filename based on param being optimized
        if a == 1 and ac == 1:
            print "Accuracy = {} for k = {} and Learn_rate = {}".format(accuracy[k],k+1,learn_rate)
            filename = 'Neural_Net_linear_ROC_Learn_rate_{}_k_{}.png'.format(learn_rate,k+1)
        elif a == 1 and l == 1:
            print "Accuracy = {} for k = {} and Activation_fcn = {}".format(accuracy[k],k+1,act_fcn)
            filename = 'Neural_Net_linear_ROC_Activation_fcn_{}_k_{}.png'.format(act_fcn,k+1)
        elif l == 1 and ac == 1:
            print "Accuracy = {} for k = {} and Alpha = {}".format(accuracy[k],k+1,alpha)
            filename = 'Neural_Net_linear_ROC_Alpha_{}_k_{}.png'.format(alpha,k+1)
        else:
            print "Accuracy = {} for k = {} and Alpha = {}, Activation_fcn = {}, Learn_rate = {}".format(accuracy[k],k+1,alpha,act_fcn,learn_rate)
            filename = 'Neural_Net_linear_ROC_Alpha_{}_Act_fcn_{}_Learn_rate_{}_k_{}.png'.format(alpha,act_fcn,learn_rate,k+1)
        
        # Generate ROC plot (all_classes, k-folds)
        title = 'ROC_Neural_Net'
        res.plot_ROC_curve(test.target,y_score,filename,title, True)
        
    return accuracy