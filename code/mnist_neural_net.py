import mnist_data as mnist
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import result_gen as res
import mnist_utils
import time

mnist_all = mnist.get_MNIST_data()
mnist_pca, _, _ = mnist.MNIST_pca(mnist_all)

mnist_train = list()
mnist_test = list()

"""
Studying the effect of samples
"""
def neural_net_samples(alpha, learn_rate, act_fcn):
    training_data_percentage = np.array([1,10,20,30,40,50,60,70,80,90,100]) * 0.01
    stratified_rand_sample = mnist_utils.MNIST_Random_Sample_Stratified()
    length = len(training_data_percentage)
    neural_net_training_time = np.zeros((length,10), dtype=float)
    neural_net_prediction_time = np.zeros((length,10), dtype=float)
    neural_net_accuracy = np.zeros((length,10), dtype=float)

    for k in xrange(length):
        for i in range(10):
            
            mnist_train_k, mnist_test_k = stratified_rand_sample.sample_train_data(training_data_percentage[k])
            
            print "Number of training samples: {}".format(len(mnist_train_k.data))
            print "Number of test samples:{}".format(len(mnist_test_k.data))
            
            # One hidden layer with 100 neurons.
            t0 = time.clock()
            mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=200, alpha=alpha,
                                solver='sgd', tol=1e-4, random_state=1,
                                learning_rate_init=learn_rate, activation=act_fcn)
            mlp.fit(mnist_train_k.data, mnist_train_k.target)
            execTime = time.clock() - t0
            neural_net_training_time[k][i] = execTime
                        
            print "Execution Time for Neural Net: {}".format(execTime)                         
            
            t0 = time.clock()          
            y_pred = mlp.predict(mnist_test_k.data)
            execTime = time.clock() - t0
            neural_net_prediction_time[k][i] = execTime
            
            accuracy_i = metrics.accuracy_score(mnist_test_k.target, y_pred)
            neural_net_accuracy[k][i] = accuracy_i
            
            print "Accuracy: {}".format(accuracy_i)
    
    np.savetxt("../experiments/expNN/neural_net_training_times_different_tss.csv", neural_net_training_time, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/expNN/neural_net_prediction_times_different_tss.csv", neural_net_prediction_time, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/expNN/neural_net_accuracy_different_tss.csv", neural_net_accuracy, delimiter=",", fmt="%.3f")
    
    plt.figure()
    plt.grid(linestyle='--')
    plt.plot(training_data_percentage * 100, np.mean(neural_net_accuracy,axis=1),'b')
    plt.title('Accuracy vs. number of samples',fontsize=14, fontweight='bold')
    plt.xlabel('Percentage of samples')
    plt.ylabel('Accuracy')
#    plt.savefig('..\Results_Plots\accuracy_vs_samples_neural_net_tuned.png', dpi = 600)
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
def neural_net_opt(alpha, act_fcn, learn_rate, train, test, k, use_pca=False):
    mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=200, alpha=alpha,
                            solver='sgd', tol=1e-4, random_state=1,
                            learning_rate_init=learn_rate, activation=act_fcn)
    t0 = time.clock()
    mlp.fit(train.data, train.target)
    execTime_train = time.clock() - t0
    
    t0 = time.clock()
    y_pred = mlp.predict(test.data)
    execTime_predict = time.clock() - t0
    
    # Metrics
    accuracy = metrics.accuracy_score(test.target, y_pred)
    y_score = mlp.predict_proba(test.data);
    
    if use_pca:       
        precision = metrics.precision_score(test.target, y_pred, average='weighted')
        recall = metrics.recall_score(test.target, y_pred, average='weighted')
        cm = metrics.confusion_matrix(test.target, y_pred)
        classes = ['0','1','2','3','4','5','6','7','8','9']
        filename = '..\Results_Plots\Neural_Net_Conf_matrix_Alpha_{}_Act_fcn_{}_Learn_rate_{}_k_{}_using_pca.png'.format(alpha,act_fcn,learn_rate,k+1)
        res.confusion_matrix_plot(cm, classes, filename)
        return accuracy, y_score, execTime_train, execTime_predict, precision, recall
    
    else:
        return accuracy, y_score, execTime_train, execTime_predict
    
"""
10-fold CV for param optimization
"""
def cross_validation_k_fold(alpha = 'default', act_fcn = 'default', learn_rate = 'default', use_pca = False):
    """
    :param alpha: Regularization term
    :param act_fcn: Activation function for the net
    :param learn_rate: Learning rate for the net
    :return: Accuracy score for each fold in the 10-fold CV
    """
    a = 0; l = 0; ac = 0
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
        if alpha == 'default': alpha = 1e-4; a = 1
        if learn_rate == 'default': learn_rate = 1e-3; l = 1
        if act_fcn == 'default': act_fcn = 'relu'; ac = 1
        
        if use_pca:
            accuracy[k], y_score, train_time[k], predict_time[k], precision[k], recall[k] = neural_net_opt(alpha, act_fcn, learn_rate, train, test, k, True)
        else:
            accuracy[k], y_score, train_time[k], predict_time[k] = neural_net_opt(alpha, act_fcn, learn_rate, train, test, k)
        
        # Filename based on param being optimized
        if a == 1 and ac == 1:
            print "Accuracy = {} for k = {} and Learn_rate = {}".format(accuracy[k],k+1,learn_rate)
            filename = '..\Results_Plots\Neural_Net_linear_ROC_Learn_rate_{}_k_{}.png'.format(learn_rate,k+1)
        elif a == 1 and l == 1:
            print "Accuracy = {} for k = {} and Activation_fcn = {}".format(accuracy[k],k+1,act_fcn)
            filename = '..\Results_Plots\Neural_Net_linear_ROC_Activation_fcn_{}_k_{}.png'.format(act_fcn,k+1)
        elif l == 1 and ac == 1:
            print "Accuracy = {} for k = {} and Alpha = {}".format(accuracy[k],k+1,alpha)
            filename = '..\Results_Plots\Neural_Net_linear_ROC_Alpha_{}_k_{}.png'.format(alpha,k+1)
        elif use_pca == False:
            print "Accuracy = {} for k = {} and Alpha = {}, Activation_fcn = {}, Learn_rate = {}".format(accuracy[k],k+1,alpha,act_fcn,learn_rate)
            filename = '..\Results_Plots\Neural_Net_linear_ROC_Alpha_{}_Act_fcn_{}_Learn_rate_{}_k_{}.png'.format(alpha,act_fcn,learn_rate,k+1)
        else:
            print "Accuracy = {} for k = {} and Alpha = {}, Activation_fcn = {}, Learn_rate = {}".format(accuracy[k],k+1,alpha,act_fcn,learn_rate)
            filename = '..\Results_Plots\Neural_Net_linear_ROC_Alpha_{}_Act_fcn_{}_Learn_rate_{}_k_{}_using_pca.png'.format(alpha,act_fcn,learn_rate,k+1)
        
        # Generate ROC plot (all_classes, k-folds)
        title = 'ROC_Neural_Net'
        res.plot_ROC_curve(test.target,y_score,filename,title, True)
    
    if use_pca:
        np.savetxt("../experiments/expNN/neural_net_precision_hypo_test.csv", precision, delimiter=",", fmt="%.3f")
        np.savetxt("../experiments/expNN/neural_net_recall_hypo_test.csv", recall, delimiter=",", fmt="%.3f")
        
    return accuracy, train_time, predict_time