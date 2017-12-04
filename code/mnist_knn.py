import mnist_utils as mnist
import mnist_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import time
import numpy as np
import result_gen as res

mnist_all = mnist.get_MNIST_data()

NCORES = 2
NFOLDS = 10


knn_training_data_class_count = np.zeros((10,10), dtype=float)
knn_test_data_class_count = np.zeros((10,10), dtype=float)

def k_fold_cross_validation_for_choosing_k_in_knn():#CV with stratification
    k_values = [1,2,3,4,5,10,15,20,25]# Differenet values of k for knn
    length = len(k_values)
    knn_training_time = np.zeros((length,10), dtype=float)
    knn_prdiction_time = np.zeros((length,10), dtype=float)
    knn_accuracy = np.zeros((length,10), dtype=float)

    cv_obj = mnist.MNIST_CV_Stratified(mnist_all) # Getting the MNIST_CV_Stratified object from mnist_all data

    for k in range(len(k_values)):
        for i in range(NFOLDS):
            mnist_train, mnist_test = cv_obj.get_train_test_split(NFOLDS, i) # Getting train, test split for ith fold

            print "Number of training samples: {}".format(len(mnist_train.data))
            print "Number of test samples:{}".format(len(mnist_test.data))

            for j in range(10):
                knn_training_data_class_count[i][j] = len(np.argwhere(mnist_train.target[:] == j).flatten())
                knn_test_data_class_count[i][j] = len(np.argwhere(mnist_test.target[:] == j).flatten())


            # Nearest neighbor
            t0 = time.clock()
            knn = KNeighborsClassifier(n_neighbors = k_values[k], n_jobs=NCORES)
            knn.fit(mnist_train.data, mnist_train.target)
            execTime = time.clock() - t0
            knn_training_time[k][i] = execTime

            print "Execution Time for Nearest Neighbor: {}".format(execTime)

            t0 = time.clock()
            y_pred = knn.predict(mnist_test.data)
            execTime = time.clock() - t0
            knn_prdiction_time[k][i] = execTime

            print "Execution Time for Nearest Neighbor Prediction: {}".format(execTime)

            print "Accuracy: {}".format(metrics.accuracy_score(mnist_test.target, y_pred))
            knn_accuracy[k][i] = metrics.accuracy_score(mnist_test.target, y_pred)


    np.savetxt("../experiments/exp5/knn_training_times_cv_different_k.csv", knn_training_time, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/exp5/knn_prdiction_times_cv_different_k.csv", knn_prdiction_time, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/exp5/knn_accuracy_varied_cv_different_k.csv", knn_accuracy, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/exp5/knn_training_data_class_counts.csv", knn_training_data_class_count, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/exp5/knn_test_data_class_counts.csv", knn_test_data_class_count, delimiter=",", fmt="%.3f")

'''
def k_fold_cross_validation_for_choosing_k_in_knn_exp1():
    k_values = [1,2,3,4,5,10,15,20,25]
    knn_training_time = np.zeros((6,9), dtype=float)
    knn_prdiction_time = np.zeros((6,9), dtype=float)
    knn_accuracy = np.zeros((6,9), dtype=float)

    for tss in xrange(6):

        mnist_train_k, mnist_test_k = mnist_data.MNIST_train_test_split_k(mnist_all, (tss + 1)*10000)

        for k in range(len(k_values)):

            print "Number of training samples: {}".format(len(mnist_train_k.data))
            print "Number of test samples:{}".format(len(mnist_test_k.data))

            # Nearest neighbor
            t0 = time.clock()
            knn = KNeighborsClassifier(n_neighbors = k_values[k], n_jobs=NCORES) #All default parameters
            knn.fit(mnist_train_k.data, mnist_train_k.target)
            execTime = time.clock() - t0
            knn_training_time[tss][k] = execTime

            print "Execution Time for Nearest Neighbor: {}".format(execTime)

            t0 = time.clock()
            y_pred = knn.predict(mnist_test_k.data)
            execTime = time.clock() - t0
            knn_prdiction_time[tss][k] = execTime

            print "Execution Time for Nearest Neighbor Prediction: {}".format(execTime)

            print "Accuracy: {}".format(metrics.accuracy_score(mnist_test_k.target, y_pred))
            knn_accuracy[tss][k] = metrics.accuracy_score(mnist_test_k.target, y_pred)


    np.savetxt("knn_training_times_varied_tss.csv", knn_training_time, delimiter=",", fmt="%.3f")
    np.savetxt("knn_prdiction_times_varied_tss.csv", knn_prdiction_time, delimiter=",", fmt="%.3f")
    np.savetxt("knn_accuracy_varied_tss.csv", knn_accuracy, delimiter=",", fmt="%.3f")


def k_fold_cross_validation_for_choosing_k_in_knn_exp3():
    k_values = [1,2,3,4,5,10,15,20,25]
    length = len(k_values)
    knn_training_time = np.zeros((length,10), dtype=float)
    knn_prdiction_time = np.zeros((length,10), dtype=float)
    knn_accuracy = np.zeros((length,10), dtype=float)

    for k in range(len(k_values)):
        for i in range(len(NFOLDS)):# Running KNN for different values of k with cv without stratification.

            mnist_train_k, mnist_test_k = mnist.MNIST_train_test_split_k_fold(mnist_all, NFOLDS, i)

            print "Number of training samples: {}".format(len(mnist_train_k.data))
            print "Number of test samples:{}".format(len(mnist_test_k.data))

            # Nearest neighbor
            t0 = time.clock()
            knn = KNeighborsClassifier(n_neighbors = k_values[k], n_jobs=NCORES)
            knn.fit(mnist_train_k.data, mnist_train_k.target)
            execTime = time.clock() - t0
            knn_training_time[k][i] = execTime

            print "Execution Time for Nearest Neighbor: {}".format(execTime)

            t0 = time.clock()
            y_pred = knn.predict(mnist_test_k.data)
            execTime = time.clock() - t0
            knn_prdiction_time[k][i] = execTime

            print "Execution Time for Nearest Neighbor Prediction: {}".format(execTime)

            print "Accuracy: {}".format(metrics.accuracy_score(mnist_test_k.target, y_pred))
            knn_accuracy[k][i] = metrics.accuracy_score(mnist_test_k.target, y_pred)


    np.savetxt("../experiments/knn_accuracy_varied_cv.csv", knn_training_time, delimiter=",", fmt="%.3f")
    np.savetxt("knn_training_times_cv.csv", knn_prdiction_time, delimiter=",", fmt="%.3f")
    np.savetxt("knn_prdiction_times_cv.csv", knn_accuracy, delimiter=",", fmt="%.3f")
'''

def knn_expt_for_varied_tss():
    # Varying the training set size [1,10,20,30,40,50,60,70,80,90,100] and
    # keeping the test set size constant last 10k data.
    training_data_percentage = np.array([1,10,20,30,40,50,60,70,80,90,100]) * 0.01
    length = len(training_data_percentage)
    knn_training_time = np.zeros((length,10), dtype=float)
    knn_prdiction_time = np.zeros((length,10), dtype=float)
    knn_accuracy = np.zeros((length,10), dtype=float)

    k = 3 #Found the value of k=3 by doing 10-fold cv with stratification

    stratified_rand_sample_obj = mnist.MNIST_Random_Sample_Stratified() # Getting the MNIST_Random_Sample_Stratified object

    for t in range(len(training_data_percentage)):
        for i in range(10):# Doing sampling 10 times for each training percentage for better generalization.
            # Getting train, test split for ith run
            mnist_train, mnist_test = \
                stratified_rand_sample_obj.sample_train_data(training_data_percentage[t])

            print "Number of training samples: {}".format(len(mnist_train.data))
            print "Number of test samples:{}".format(len(mnist_test.data))

            # Nearest neighbor
            t0 = time.clock()
            knn = KNeighborsClassifier(n_neighbors = k, n_jobs=NCORES)
            knn.fit(mnist_train.data, mnist_train.target)
            execTime = time.clock() - t0
            knn_training_time[t][i] = execTime

            print "Execution Time for Nearest Neighbor: {}".format(execTime)

            t0 = time.clock()
            y_pred = knn.predict(mnist_test.data)
            execTime = time.clock() - t0
            knn_prdiction_time[t][i] = execTime

            print "Execution Time for Nearest Neighbor Prediction: {}".format(execTime)

            print "Accuracy: {}".format(metrics.accuracy_score(mnist_test.target, y_pred))
            knn_accuracy[t][i] = metrics.accuracy_score(mnist_test.target, y_pred)


    np.savetxt("../experiments/exp6/knn_training_times_different_tss.csv", knn_training_time, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/exp6/knn_prdiction_times_different_tss.csv", knn_prdiction_time, delimiter=",", fmt="%.3f")
    np.savetxt("../experiments/exp6/knn_accuracy_different_tss.csv", knn_accuracy, delimiter=",", fmt="%.3f")


def k_fold_cross_validation_knn(use_pca = False):

    accuracy = np.zeros(10)
    precision = np.zeros(10)
    recall = np.zeros(10)
    train_time = np.zeros(10)
    predict_time = np.zeros(10)

    k = 3 # Value of k chosen through CV.

    cv_obj = mnist.MNIST_CV_Stratified(mnist_all)

    for i in xrange(NFOLDS):
        train, test = cv_obj.get_train_test_split(NFOLDS, i)

        if use_pca:
            # Generating PCA features on the training data and then
            # applying those features to the test data
            pc, mean = mnist.MNIST_pca(train)
            train_pca = mnist.PCA_transform(train, pc, mean)
            test_pca = mnist.PCA_transform(test, pc, mean)
            accuracy[i], y_score, train_time[i], predict_time[i], precision[i], recall[i] = \
                knn_k(k,train_pca,test_pca,i,True)
        else:
            accuracy[i], y_score, train_time[i], predict_time[i] = knn_k(k,train,test,i)

        print "Accuracy = {} for fold = {} and K = {}".format(accuracy[i],i+1,k)

        if not use_pca:
            filename = '../experiments/expKNN/KNN_ROC_K_{}_fold_{}.pdf'.format(k,i+1)
        else:
            filename = '../experiments/expKNN/KNN_ROC_K_{}_fold_{}_using_pca.pdf'.format(k,i+1)

        title = 'ROC_KNN'
        res.plot_ROC_curve(test.target,y_score,filename,title, True)

    if use_pca:
        np.savetxt("../experiments/expKNN/knn_precision_hypo_test.csv", precision, delimiter=",", fmt="%.3f")
        np.savetxt("../experiments/expKNN/knn_recall_hypo_test.csv", recall, delimiter=",", fmt="%.3f")

        np.savetxt("../experiments/expKNN/knn_accuracy_hypo_test_with_pca.csv", accuracy, delimiter=",", fmt="%.3f")
        np.savetxt("../experiments/expKNN/knn_train_time_hypo_test_with_pca.csv", train_time, delimiter=",", fmt="%.3f")
        np.savetxt("../experiments/expKNN/knn_predict_time_hypo_test_with_pca.csv", predict_time, delimiter=",", fmt="%.3f")

    else:
        #return accuracy, train_time, predict_time
        np.savetxt("../experiments/expKNN/knn_accuracy_hypo_test_without_pca.csv", accuracy, delimiter=",", fmt="%.3f")
        np.savetxt("../experiments/expKNN/knn_train_time_hypo_test_without_pca.csv", train_time, delimiter=",", fmt="%.3f")
        np.savetxt("../experiments/expKNN/knn_predict_time_hypo_test_without_pca.csv", predict_time, delimiter=",", fmt="%.3f")



def knn_k(k, train, test, fold, use_pca=False):

    knn = KNeighborsClassifier(n_neighbors = k, n_jobs=NCORES)

    t0 = time.clock()
    knn.fit(train.data, train.target)
    execTime_train = time.clock() - t0

    t0 = time.clock()
    y_pred = knn.predict(test.data)
    execTime_predict = time.clock() - t0

    # Metrics
    accuracy = metrics.accuracy_score(test.target, y_pred)
    y_score = knn.predict_proba(test.data)

    if use_pca:
        precision = metrics.precision_score(test.target, y_pred, average='weighted')
        recall = metrics.recall_score(test.target, y_pred, average='weighted')
        cm = metrics.confusion_matrix(test.target, y_pred)
        classes = ['0','1','2','3','4','5','6','7','8','9']
        filename = '../experiments/expKNN/KNN_Conf_matrix_k_{}_fold_{}_using_pca.pdf'.format(k,fold+1)
        res.confusion_matrix_plot(cm, classes, filename)
        return accuracy, y_score, execTime_train, execTime_predict, precision, recall

    else:
        return accuracy, y_score, execTime_train, execTime_predict


def main():
    #k_fold_cross_validation_for_choosing_k_in_knn()
    #knn_expt_for_varied_tss()
    k_fold_cross_validation_knn(True) # Using PCA

if __name__=="__main__":
    main()