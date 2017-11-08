import neural_net_experiments
import svm_experiments
import mnist_knn

"""Neural Network (Parameter tuning)"""
neural_net_experiments.k_fold_CV_regularization_Neural_Net()
neural_net_experiments.k_fold_CV_Activation_fcn_Neural_Net()
neural_net_experiments.k_fold_CV_Learn_rate_Neural_Net()

"""Support Vector Machines(Parameter tuning)"""
svm_experiments.k_fold_CV_regularization_SVM()

"""K-Nearest Neighbours (Parameter tuning)"""
mnist_knn.k_fold_cross_validation_for_choosing_k_in_knn()
#mnist_knn.k_fold_cross_validation_for_choosing_k_in_knn_exp1()
#mnist_knn.k_fold_cross_validation_for_choosing_k_in_knn_exp3()
#mnist_knn.knn_expt_for_varied_tss()