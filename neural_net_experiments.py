import mnist_neural_net
import numpy as np
import matplotlib.pyplot as plt

## Studying the effect of samples
#mnist_neural_net.neural_net_samples()
#
## optimizing the size of hidden layers
#mnist_neural_net.neural_net_layers_size()

# Optimizing the regulairtion term (alpha) 
alpha = np.logspace(-4,2,7)
mean_accuracy_alp = np.array([])
var_accuracy_alp = np.array([])
k = 0
accuracy_alp = np.zeros((np.size(alpha),10))

for alp in alpha:    
    # Run cross-validation
    accuracy_alp[k,:] = mnist_neural_net.cross_validation_k_fold(alpha = alp)       
    
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

# Studying effect of activation function 
act_fcns = ['identity','logistic','tanh', 'relu']
mean_accuracy_act_fcn = np.array([])
var_accuracy_act_fcn = np.array([])
k = 0
accuracy_act_fcn = np.zeros((len(act_fcns),10))

for func in act_fcns:
    # Run cross-validation
    accuracy_act_fcn[k,:] = mnist_neural_net.cross_validation_k_fold(act_fcn=func)       
    
    # Mean and Variance of accuracy metric
    mean_accuracy_act_fcn = np.append(mean_accuracy_act_fcn,np.mean(accuracy_act_fcn[k,:]))    
    var_accuracy_act_fcn = np.append(var_accuracy_act_fcn,np.var(accuracy_act_fcn[k,:]))
    
    plt.figure()
    plt.plot(accuracy_act_fcn[k,:])
    plt.xlabel('k-fold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. CV fold for activation_func = {}'.format(func))
    plt.show()   
    k += 1

# Optimizing learning rate
learn_rate = np.arange(0.001, 0.1, 0.001)
mean_accuracy_learn_rate = np.array([])
var_accuracy_learn_rate = np.array([])
k = 0
accuracy_learn_rate = np.zeros((np.size(learn_rate),10))

for learn_r in learn_rate:
    # Run cross-validation
    accuracy_learn_rate[k,:] = mnist_neural_net.cross_validation_k_fold(learn_rate=learn_r)       
    
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