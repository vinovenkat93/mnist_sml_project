import neural_net_experiments
import svm_experiments
import mnist_knn

"""Neural Network (Parameter tuning)"""
#neural_net_experiments.k_fold_CV_regularization_Neural_Net()
#neural_net_experiments.k_fold_CV_Activation_fcn_Neural_Net()
#neural_net_experiments.k_fold_CV_Learn_rate_Neural_Net()
#neural_net_experiments.neural_net_tss()

"""Support Vector Machines(Parameter tuning)"""
#svm_experiments.k_fold_CV_regularization_SVM()

"""K-Nearest Neighbours (Parameter tuning)"""
#mnist_knn.k_fold_cross_validation_for_choosing_k_in_knn()


"""Samples vs. accuracy experiments"""
#neural_net_experiments.neural_net_tss()
#svm_experiments.svm_tss()
#mnist_knn.knn_expt_for_varied_tss()

"""Hypothesis testing"""
neural_net_experiments.k_fold_CV_hypothesis_test()
svm_experiments.k_fold_CV_hypothesis_test()
#neural_net_experiments.k_fold_CV_hypothesis_test_without_PCA()
