import mnist_svm
import numpy as np
import matplotlib.pyplot as plt

# Effect of samples
#mnist_svm.svm_linear_samples()
#mnist_svm.svm_rbf_samples()

# k- fold cross_validation (10-fold)
C = np.logspace(-2,2,num=5)
C = np.append(C,np.array([2,5,20,40,50,60,70,80,90]))
for i in C:
    mean_accuracy = np.array([])
    var_accuracy = np.array([])
    
    # Run cross-validation
    accuracy = mnist_svm.cross_validation_k_fold(i)       
    
    mean_accuracy = np.append(mean_accuracy,np.mean(accuracy))    
    var_accuracy = np.append(var_accuracy,np.var(accuracy))
    
    plt.figure()
    plt.plot(accuracy)
    plt.show()
    plt.xlabel('k-fold')
    plt.ylabel('Accuracy')
    plt.title('Avvuracy vs. CV fold for C = {}'.format(i))
# Accuracy/MSE for comparing regularization parameter (in SVM alone, KNN alone , etc,.)

# ROC curves for comparing models (SVM vs. KNN vs. Neural)


