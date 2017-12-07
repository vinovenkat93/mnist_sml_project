import numpy as np
import matplotlib.pyplot as plt
import hypothesis_testing


# Effect of samples
def plot_results(file_name):
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    #values = np.loadtxt(file_name , delimiter=',')
    values = np.loadtxt(file_name, delimiter=",")[2,:]
    plt.plot(x,values)
    '''
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    xmin -= 0.1
    xmax += 0.1
    ymin -= 0.01
    ymax += 0.01
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    '''
    plt.xticks(x, x)

    plt.xlabel("k-fold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy(10-fold CV)")
    plt.show()
    #plt.savefig("plot_1.pdf", bbox_inches='tight')


def plot_accuracy(accuracy_file_name):
    accuracy_values = np.loadtxt(accuracy_file_name , delimiter=',')
    '''
    accuracy_values_2 = np.loadtxt(accuracy_file_name_1 , delimiter=',')
    accuracy_values = np.zeros((accuracy_values_1.shape[0] + accuracy_values_2.shape[0], accuracy_values_1.shape[1]))
    accuracy_values[0,:] = accuracy_values_1[0,:]
    accuracy_values[1:4,:] = accuracy_values_2
    accuracy_values[4:] = accuracy_values_1[1:]
    '''
    x = np.arange(accuracy_values.shape[0])
    n = accuracy_values.shape[1]
    x_labels = np.array([1,10,20,30,40,50,60,70,80,90,100]) #np.array([1,2,3,4,5,10,15,20,25])

    plt.errorbar(x, accuracy_values.mean(axis=1), yerr=accuracy_values.std(axis=1)/np.sqrt(n),
                 label='Accuracy')

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    print "ymin, ymax", ymin, ymax
    xmin -= 0.1
    xmax += 0.1
    ymin -= 0.01
    ymax += 0.01
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.xticks(x, x_labels)
    plt.legend(loc='upper right')
    plt.xlabel("Training Data Percentage")
    plt.ylabel("Accuracy")
    plt.savefig("../experiments/exp6/plot_accuracy_vs_tss_with_fixed_k.pdf", bbox_inches='tight')
    '''
    plt.clf()
    plt.errorbar(x, training_values.mean(axis=1), yerr=training_values.std(axis=1)/np.sqrt(n),
                 label='Training times')

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    xmin -= 0.1
    xmax += 0.1
    ymin -= 0.01
    ymax += 0.01
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.xticks(x, x_labels)
    plt.legend(loc='upper right')
    plt.xlabel("K")
    plt.ylabel("Training times")
    plt.xticks(x, x_labels)
    plt.savefig("plot_training_times_vs_k.pdf", bbox_inches='tight')

    plt.clf()
    plt.errorbar(x, prediction_values.mean(axis=1), yerr=prediction_values.std(axis=1)/np.sqrt(n),
                 label='Prediction times')

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    xmin -= 0.1
    xmax += 0.1
    ymin -= 0.01
    ymax += 0.01
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.xticks(x, x_labels)
    plt.legend(loc='upper right')
    plt.xlabel("K")
    plt.ylabel("Prediction times")
    plt.xticks(x, x_labels)
    plt.savefig("plot_prediction_times_vs_k.pdf", bbox_inches='tight')
    '''



def plot_class_numbers(training_data_file_name, test_data_file_name):
    training_data = np.loadtxt(training_data_file_name , delimiter=',')
    test_data = np.loadtxt(test_data_file_name , delimiter=',')

    x = np.arange(training_data.shape[0])
    n = training_data.shape[1]
    x_labels = np.array([0,1,2,3,4,5,6,7,8,9])

    plt.errorbar(x, training_data.mean(axis=0), yerr=training_data.std(axis=0)/np.sqrt(n),
                 label='Class Count')

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    print "ymin, ymax", ymin, ymax
    xmin -= 0.1
    xmax += 0.1
    ymin -= 0.01
    ymax += 0.01
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.xticks(x, x_labels)
    plt.legend(loc='upper right')
    plt.xlabel("Class Labels")
    plt.ylabel("Class Count")
    plt.savefig("plot_training_class_count.pdf", bbox_inches='tight')

    plt.clf()
    plt.errorbar(x, test_data.mean(axis=0), yerr=test_data.std(axis=0)/np.sqrt(n),
                 label='Class count')
    plt.xticks(x, x_labels)
    plt.legend(loc='upper right')
    plt.xlabel("Class Labels")
    plt.ylabel("Class Count")
    plt.savefig("plot_test_class_count.pdf", bbox_inches='tight')


def hypo_SVM_KNN_without_PCA():
    svm_accuracy = np.loadtxt("../experiments/expSVM/svm_accuracy_vs_C.csv", delimiter=",")[11,:]
    knn_accuracy = np.loadtxt("../experiments/exp5/knn_accuracy_varied_cv_different_k.csv", delimiter=",")[2,:]

    x, x_alpha = hypothesis_testing.hypothesis_testing(svm_accuracy,knn_accuracy)

    print 'hypo_SVM_KNN_without_PCA: x=%f, x_alpha=%f' %(x, x_alpha)


def hypo_NN_KNN_without_PCA():
    nn_accuracy = np.loadtxt("../experiments/expNN/neural_net_accuracy_hypo_test_without_pca.csv", delimiter=",")
    knn_accuracy = np.loadtxt("../experiments/exp5/knn_accuracy_varied_cv_different_k.csv", delimiter=",")[2,:]

    x, x_alpha = hypothesis_testing.hypothesis_testing(nn_accuracy,knn_accuracy)
    print 'hypo_NN_KNN_without_PCA: x=%f, x_alpha=%f' %(x, x_alpha)


def hypo_SVM_KNN_PCA():
    svm_accuracy = np.loadtxt("../experiments/expSVM/svm_accuracy_hypo_test.csv", delimiter=",")
    knn_accuracy = np.loadtxt("../experiments/expKNN/knn_accuracy_hypo_test_with_pca.csv", delimiter=",")

    x, x_alpha = hypothesis_testing.hypothesis_testing(svm_accuracy,knn_accuracy)
    print 'hypo_SVM_KNN_PCA: x=%f, x_alpha=%f' %(x, x_alpha)


def hypo_NN_KNN_PCA():
    nn_accuracy = np.loadtxt("../experiments/expNN/neural_net_accuracy_hypo_test.csv", delimiter=",")
    knn_accuracy = np.loadtxt("../experiments/expKNN/knn_accuracy_hypo_test_with_pca.csv", delimiter=",")

    x, x_alpha = hypothesis_testing.hypothesis_testing(nn_accuracy,knn_accuracy)
    print 'hypo_NN_KNN_PCA: x=%f, x_alpha=%f' %(x, x_alpha)


def hypo_KNN_PCA_vs_without_PCA():
    knn_accuracy_PCA = np.loadtxt("../experiments/expKNN/knn_accuracy_hypo_test_with_pca.csv", delimiter=",")
    knn_accuracy_without_PCA = np.loadtxt("../experiments/exp5/knn_accuracy_varied_cv_different_k.csv", delimiter=",")[2,:]

    x, x_alpha = hypothesis_testing.hypothesis_testing(knn_accuracy_PCA, knn_accuracy_without_PCA)
    print 'hypo_KNN_PCA_vs_without_PCA: x=%f, x_alpha=%f' %(x, x_alpha)


def main():
    #plot_accuracy("./experiments/knn_accuracy_varied_cv.csv","./experiments/knn_training_times_cv.csv",
    #              "./experiments/knn_prdiction_times_cv.csv")

    #plot_accuracy("./experiments/knn_accuracy_varied_cv.csv","./experiments/knn_accuracy_varied_cv_k234.csv", "./experiments/knn_training_times_cv.csv",
    #             "./experiments/knn_prdiction_times_cv.csv")

    #plot_accuracy("../experiments/exp6/knn_accuracy_different_tss.csv")
    plot_results("../experiments/exp5/knn_accuracy_varied_cv_different_k.csv")


if __name__=="__main__":
    main()