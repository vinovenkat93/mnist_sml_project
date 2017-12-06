import neural_net_experiments
import svm_experiments
import mnist_knn
import hypothesis_testing
import mnist_utils
import result_gen
import knn_plot_gen as hypo

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

''' 
Data generation for each algorithm
'''
#neural_net_experiments.k_fold_CV_hypothesis_test()
#svm_experiments.k_fold_CV_hypothesis_test()
#neural_net_experiments.k_fold_CV_hypothesis_test_without_PCA()

''' 
Using data to perform hypothesis testing
'''
#hypo_1 = result_gen.hypo_SVM_NN_PCA()
#hypo_2 = result_gen.hypo_SVM_NN_without_PCA()
#hypo_3 = result_gen.hypo_SVM_PCA_vs_without_PCA()
#hypo_4 = result_gen.hypo_NN_PCA_vs_without_PCA()
hypo.hypo_SVM_KNN_without_PCA()
hypo.hypo_NN_KNN_without_PCA()
hypo.hypo_SVM_KNN_PCA()
hypo.hypo_NN_KNN_PCA()
hypo.hypo_KNN_PCA_vs_without_PCA()
