import mnist_neural_net
import numpy as np
import matplotlib.pyplot as plt

def k_fold_CV_regularization_Neural_Net():
    """
    Optimizing the regulairtion term (alpha) 
    """
    alpha = np.logspace(-4,4,9)
    mean_accuracy_alp = np.array([])
    var_accuracy_alp = np.array([])
    k = 0
    accuracy_alp = np.zeros((np.size(alpha),10))
    train_time = np.zeros((np.size(alpha),10))
    predict_time = np.zeros((np.size(alpha),10))
    
    for alp in alpha:    
        # Run cross-validation
        accuracy_alp[k,:], train_time[k,:], predict_time[k,:] = mnist_neural_net.cross_validation_k_fold(alpha = alp)       
        
        # Mean and Variance of accuracy metric
        mean_accuracy_alp = np.append(mean_accuracy_alp,np.mean(accuracy_alp[k,:]))    
        var_accuracy_alp = np.append(var_accuracy_alp,np.var(accuracy_alp[k,:]))
        
        plt.figure()
        plt.plot(accuracy_alp[k,:])
        plt.xlabel('k-fold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. CV fold for alpha = {}'.format(alp))
        plt.show()   
        k += 1
    
    np.savetxt("../experiments/expNN/neural_net_accuracy_vs_alpha.csv", accuracy_alp, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/expNN/neural_net_train_time_alpha.csv", train_time, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/expNN/neural_net_predict_time_alpha.csv", predict_time, delimiter=",", fmt="%.3f")
    
    # Mean_accuracy plots
    plt.figure()
    plt.plot(mean_accuracy_alp)
    plt.grid(linestyle='--')
    plt.xscale('log')
    plt.gca().set(xlabel=r'$\alpha$ (regularization)', ylabel='Mean Accuracy (10-fold CV)')
    plt.title(r'$\alpha$ (regularization) optimization using the 10-fold CV') 
    plt.savefig('..\Results_Plots\Neural_net_alpha_mean_acc.png', dpi = 600)
    plt.show()  
    
    plt.figure()
    plt.plot(alpha, var_accuracy_alp)
    plt.grid(linestyle='--')
    plt.xscale('log')
    plt.gca().set(xlabel=r'$\alpha$ (regularization)', ylabel='Variance Accuracy (10-fold CV)')
    plt.title(r'$\alpha$ (regularization) optimization using the 10-fold CV') 
    plt.savefig('..\Results_Plots\Neural_net_alpha_var_acc.png', dpi = 600)
    plt.show() 

def k_fold_CV_Activation_fcn_Neural_Net():
    """
    Studying effect of activation function 
    """
    act_fcns = ['logistic','tanh', 'relu']
    mean_accuracy_act_fcn = np.array([])
    var_accuracy_act_fcn = np.array([])
    k = 0
    accuracy_act_fcn = np.zeros((len(act_fcns),10))
    train_time = np.zeros((len(act_fcns),10))
    predict_time = np.zeros((len(act_fcns),10))
    
    for func in act_fcns:
        # Run cross-validation
        accuracy_act_fcn[k,:], train_time[k,:], predict_time[k,:] = mnist_neural_net.cross_validation_k_fold(act_fcn=func)       
        
        # Mean and Variance of accuracy metric
        mean_accuracy_act_fcn = np.append(mean_accuracy_act_fcn,np.mean(accuracy_act_fcn[k,:]))    
        var_accuracy_act_fcn = np.append(var_accuracy_act_fcn,np.var(accuracy_act_fcn[k,:]))
        
        plt.figure()
        plt.plot(accuracy_act_fcn[k,:])
        plt.xlabel('k-fold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. CV fold for activation_func: {}'.format(func))
        plt.show()   
        k += 1
    
    np.savetxt("../experiments/expNN/neural_net_accuracy_vs_activation_fcn.csv", accuracy_act_fcn, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/expNN/neural_net_train_time_activation_fcn.csv", train_time, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/expNN/neural_net_predict_time_activation_fcn.csv", predict_time, delimiter=",", fmt="%.3f")
    
    # Mean_accuracy plots
    fig, ax = plt.subplots(1,1)
    ax.set_xticks(np.arange(0,3,1))
    ax.set_xticklabels(act_fcns, rotation=20)
    ax.plot(mean_accuracy_act_fcn)
    ax.grid(linestyle='--')
    ax.set_title('Choosing activation function using the 10-fold CV')
    ax.set_ylabel('Mean accuracy') 
    plt.savefig('..\Results_Plots\Neural_net_act_fcn_mean_acc.png', dpi = 600)
    plt.show()  
    
    # Variance_accuracy plots
    fig, ax = plt.subplots(1,1)
    ax.set_xticks(np.arange(0,3,1))
    ax.set_xticklabels(act_fcns, rotation=20)
    ax.plot(var_accuracy_act_fcn)
    ax.grid(linestyle='--')
    ax.set_title('Choosing activation function using the 10-fold CV')
    ax.set_ylabel('Variance accuracy') 
    plt.savefig('..\Results_Plots\Neural_net_act_fcn_var_acc.png', dpi = 600)
    plt.show()  
    
def k_fold_CV_Learn_rate_Neural_Net():
    """
    Optimizing learning rate
    """
    learn_rate = np.arange(0.0001, 0.0021, 0.0001)
    mean_accuracy_learn_rate = np.array([])
    var_accuracy_learn_rate = np.array([])
    k = 0
    accuracy_learn_rate = np.zeros((np.size(learn_rate),10))
    train_time = np.zeros((np.size(learn_rate),10))
    predict_time = np.zeros((np.size(learn_rate),10))
    
    for learn_r in learn_rate:
        # Run cross-validation
        accuracy_learn_rate[k,:], train_time[k,:], predict_time[k,:] = mnist_neural_net.cross_validation_k_fold(learn_rate=learn_r)       
        
        # Mean and Variance of accuracy metric
        mean_accuracy_learn_rate = np.append(mean_accuracy_learn_rate,np.mean(accuracy_learn_rate[k,:]))    
        var_accuracy_learn_rate = np.append(var_accuracy_learn_rate,np.var(accuracy_learn_rate[k,:]))
        
        plt.figure()
        plt.plot(accuracy_learn_rate[k,:])
        plt.xlabel('k-fold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. CV fold for learning_rate = {}'.format(learn_r))
        plt.show()   
        k += 1
     
    np.savetxt("../experiments/expNN/neural_net_accuracy_vs_learn_rate.csv", accuracy_learn_rate, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/expNN/neural_net_train_time_learn_rate.csv", train_time, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/expNN/neural_net_predict_time_learn_rate.csv", predict_time, delimiter=",", fmt="%.3f")
    
    # Mean_accuracy plots
    plt.figure()
    plt.plot(learn_rate, mean_accuracy_learn_rate)
    plt.grid(linestyle='--')
    plt.gca().set(xlabel=r'Learning rate ($\eta$)', ylabel='Mean Accuracy (10-fold CV)')
    plt.title(r'Learning rate ($\eta$) optimization using the 10-fold CV') 
    plt.savefig('..\Results_Plots\Neural_net_learn_rate_mean_acc.png', dpi = 600)
    plt.show()  
    
    plt.figure()
    plt.plot(learn_rate, var_accuracy_learn_rate)
    plt.grid(linestyle='--')
    plt.gca().set(xlabel=r'Learning rate ($\eta$)', ylabel='Mean Accuracy (10-fold CV)')
    plt.title(r'Learning rate ($\eta$) optimization using the 10-fold CV') 
    plt.savefig('..\Results_Plots\Neural_net_learn_rate_var_acc.png', dpi = 600)
    plt.show() 

"""
Studying the effect of samples
"""
#mnist_neural_net.neural_net_samples(alpha)

"""
Optimizing the size of hidden layers
"""
#mnist_neural_net.neural_net_layers_size()