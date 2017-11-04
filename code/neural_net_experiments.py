import mnist_neural_net
import numpy as np
import time
import matplotlib.pyplot as plt

"""
Studying the effect of samples
"""
mnist_neural_net.neural_net_samples()

"""
Optimizing the size of hidden layers
"""
mnist_neural_net.neural_net_layers_size()

"""
Optimizing the regulairtion term (alpha) 
"""
alpha = np.logspace(-4,4,9)
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

# Mean_accuracy plots
plt.figure()
plt.plot(alpha, mean_accuracy_alp)
plt.grid(linestyle='--')
plt.xscale('log')
plt.gca().set(xlabel=r'$\alpha$ (regularization)', ylabel='Mean Accuracy (10-fold CV)')
plt.title(r'$\alpha$ (regularization) optimization using the 10-fold CV') 
plt.savefig('.\Results_Plots\Neural_net_alpha_mean_acc.png', dpi = 600)
plt.show()  

plt.figure()
plt.plot(alpha, var_accuracy_alp)
plt.grid(linestyle='--')
plt.xscale('log')
plt.gca().set(xlabel=r'$\alpha$ (regularization)', ylabel='Variance Accuracy (10-fold CV)')
plt.title(r'$\alpha$ (regularization) optimization using the 10-fold CV') 
plt.savefig('.\Results_Plots\Neural_net_alpha_var_acc.png', dpi = 600)
plt.show() 

"""
Studying effect of activation function 
"""
act_fcns = ['logistic','tanh', 'relu']
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
    plt.title('Accuracy vs. CV fold for activation_func: {}'.format(func))
    plt.show()   
    k += 1

# Mean_accuracy plots
fig, ax = plt.subplots(1,1)
ax.set_xticks(np.arange(0,3,1))
ax.set_xticklabels(act_fcns, rotation=20)
ax.plot(mean_accuracy_act_fcn)
ax.grid(linestyle='--')
ax.set_title('Choosing activation function using the 10-fold CV')
ax.set_ylabel('Mean accuracy') 
plt.savefig('.\Results_Plots\Neural_net_act_fcn_mean_acc.png', dpi = 600)
plt.show()  

# Variance_accuracy plots
fig, ax = plt.subplots(1,1)
ax.set_xticks(np.arange(0,3,1))
ax.set_xticklabels(act_fcns, rotation=20)
ax.plot(var_accuracy_act_fcn)
ax.grid(linestyle='--')
ax.set_title('Choosing activation function using the 10-fold CV')
ax.set_ylabel('Variance accuracy') 
plt.savefig('.\Results_Plots\Neural_net_act_fcn_var_acc.png', dpi = 600)
plt.show()  

"""
Optimizing learning rate
"""
learn_rate = np.arange(0.0001, 0.0021, 0.0001)
mean_accuracy_learn_rate = np.array([])
var_accuracy_learn_rate = np.array([])
k = 0
accuracy_learn_rate = np.zeros((np.size(learn_rate),10))
execTime = np.zeros(np.size(learn_rate))

for learn_r in learn_rate:
    # Run cross-validation
    t0 = time.clock()
    accuracy_learn_rate[k,:] = mnist_neural_net.cross_validation_k_fold(learn_rate=learn_r)       
    execTime[k] = time.clock() - t0
    
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
    
# Mean_accuracy plots
plt.figure()
plt.plot(learn_rate, mean_accuracy_learn_rate)
plt.grid(linestyle='--')
plt.gca().set(xlabel=r'Learning rate ($\eta$)', ylabel='Mean Accuracy (10-fold CV)')
plt.title(r'Learning rate ($\eta$) optimization using the 10-fold CV') 
plt.savefig('.\Results_Plots\Neural_net_learn_rate_mean_acc.png', dpi = 600)
plt.show()  

plt.figure()
plt.plot(learn_rate, var_accuracy_learn_rate)
plt.grid(linestyle='--')
plt.gca().set(xlabel=r'Learning rate ($\eta$)', ylabel='Mean Accuracy (10-fold CV)')
plt.title(r'Learning rate ($\eta$) optimization using the 10-fold CV') 
plt.savefig('.\Results_Plots\Neural_net_learn_rate_var_acc.png', dpi = 600)
plt.show() 