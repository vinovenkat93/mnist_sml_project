import mnist_svm
import numpy as np
import matplotlib.pyplot as plt

def k_fold_CV_regularization_SVM():
    """
    Optimizing the regulairtion term (C) 
    """
    C = np.logspace(-2,2,num=5)
    C = np.append(C,np.array([2,5,20,40,50,60,70,80,90]))
    
    mean_accuracy = np.array([])
    var_accuracy = np.array([])
    k = 0
    accuracy = np.zeros((np.size(C),10))
    train_time = np.zeros((np.size(C),10))
    predict_time = np.zeros((np.size(C),10))
    
    for i in C:
        
        # Run cross-validation
        accuracy[k,:], train_time[k,:], predict_time[k,:] = mnist_svm.cross_validation_k_fold(i)       
        
        mean_accuracy = np.append(mean_accuracy,np.mean(accuracy[k,:]))    
        var_accuracy = np.append(var_accuracy,np.var(accuracy[k,:]))
        
        plt.figure()
        plt.plot(accuracy[k,:])
        plt.xlabel('k-fold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. CV fold for C = {}'.format(i))
        plt.show()
        
        k += 1
     
    np.savetxt("../experiments/expSVM/svm_accuracy_vs_C.csv", accuracy, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/expSVM/svm_train_time_C.csv", train_time, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/expSVM/svm_predict_time_C.csv", predict_time, delimiter=",", fmt="%.3f")
    
    # Mean_accuracy plots
    mean_accuracy_ord = np.array([])
    mean_accuracy_ord = np.concatenate((mean_accuracy[:3],mean_accuracy[5:7],mean_accuracy[7:]))
    mean_accuracy_ord = np.insert(mean_accuracy_ord, 5,mean_accuracy[3])
    mean_accuracy_ord = np.insert(mean_accuracy_ord, 13,mean_accuracy[4])
    C_ord = np.array([])
    C_ord = np.concatenate((C[:3],C[5:7],C[7:]))
    C_ord = np.insert(C_ord, 5,C[3])
    C_ord = np.insert(C_ord, 13,C[4])
    plt.figure()
    plt.grid(linestyle='--')
    plt.plot(C_ord,mean_accuracy_ord)
    plt.gca().set(xlabel='C (regularization)', ylabel='Mean Accuracy (10-fold CV)')
    plt.title('C (regularization) optimization using the 10-fold CV', fontweight='bold')    
    plt.savefig('..\Results_Plots\SVM_regularization_C_mean_acc.png', dpi = 600)
    plt.show()
    
    var_accuracy_ord = np.array([])
    var_accuracy_ord = np.concatenate((var_accuracy[:3],var_accuracy[5:7],var_accuracy[7:]))
    var_accuracy_ord = np.insert(var_accuracy_ord, 5,var_accuracy[3])
    var_accuracy_ord = np.insert(var_accuracy_ord, 13,var_accuracy[4])
    plt.figure()
    plt.plot(C_ord, var_accuracy_ord)
    plt.grid(linestyle='--')
    plt.gca().set(xlabel='C (regularization)', ylabel='Variance Accuracy (10-fold CV)')
    plt.title('C (regularization) optimization using the 10-fold CV', fontweight='bold')    
    plt.savefig('..\Results_Plots\SVM_regularization_C_var_acc.png', dpi = 600)
    plt.show() 
  
# TODO: Accuracy/MSE for optimizing regularization parameter (KNN, Neural net)
# TODO: ROC curves for comparing models (SVM vs. KNN vs. Neural)

"""
Studying the effect of samples
"""
def svm_tss():
    C = 70
    mnist_svm.svm_linear_samples(C)

#mnist_svm.svm_rbf_samples(C)

"""
K- fold CV for hypothesis testing
"""
def k_fold_CV_hypothesis_test():
    C = 70
    accuracy, train_time, predict_time = mnist_svm.cross_validation_k_fold(C = C, use_pca=True)
    
    np.savetxt("../experiments/expSVM/svm_accuracy_hypo_test.csv", accuracy, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/expSVM/svm_train_time_hypo_test.csv", train_time, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/expSVM/svm_predict_time_hypo_test.csv", predict_time, delimiter=",", fmt="%.3f")
