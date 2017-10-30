import mnist_svm
import numpy as np
import matplotlib.pyplot as plt

# Effect of samples
#mnist_svm.svm_linear_samples()
#mnist_svm.svm_rbf_samples()

# k- fold cross_validation (10-fold)
C = np.logspace(-2,2,num=5)
C = np.append(C,np.array([2,5,20,40,50,60,70,80,90]))

mean_accuracy = np.array([])
var_accuracy = np.array([])
k = 0
accuracy = np.zeros((np.size(C),10))

for i in C:
    
    # Run cross-validation
    accuracy[k,:] = mnist_svm.cross_validation_k_fold(i)       
    
    mean_accuracy = np.append(mean_accuracy,np.mean(accuracy[k,:]))    
    var_accuracy = np.append(var_accuracy,np.var(accuracy[k,:]))
    
    plt.figure()
    plt.plot(accuracy[k,:])
    plt.xlabel('k-fold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. CV fold for C = {}'.format(i))
    plt.show()
    
    k += 1
 
# Mean_accuracy plots
mean_accuracy_ord = np.array([])
mean_accuracy_ord = np.concatenate((mean_accuracy[:3],mean_accuracy[5:7],mean_accuracy[7:]))
mean_accuracy_ord = np.insert(mean_accuracy_ord, 5,mean_accuracy[3])
mean_accuracy_ord = np.insert(mean_accuracy_ord, 13,mean_accuracy[4])
plt.figure()
plt.plot(mean_accuracy_ord)
plt.gca().set(xlabel='C (regularization)', ylabel='Mean Accuracy (10-fold CV)')
plt.title('C (regularization) optimization using the 10-fold CV', fontweight='bold')    
    
# TODO: Accuracy/MSE for optimizing regularization parameter (KNN, Neural net)
# TODO: ROC curves for comparing models (SVM vs. KNN vs. Neural)




