import mnist_utils as mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import time
import numpy as np

mnist_all = mnist.get_MNIST_data()

NCORES = 6
NFOLDS = 10

knn_training_time = np.zeros((9,10), dtype=float)
knn_prdiction_time = np.zeros((9,10), dtype=float)
knn_accuracy = np.zeros((9,10), dtype=float)

knn_training_data_class_count = np.zeros((10,10), dtype=float)
knn_test_data_class_count = np.zeros((10,10), dtype=float)

def k_fold_cross_validation_for_choosing_k_in_knn():#CV with stratification
    k_values = [1,2,3,4,5,10,15,20,25]# Differenet values of k for knn

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


def k_fold_cross_validation_for_choosing_k_in_knn_exp1():
    k_values = [1,2,3,4,5,10,15,20,25]
    for tss in xrange(6):

        mnist_train_k, mnist_test_k = mnist.MNIST_train_test_split_k(mnist_all, (tss + 1)*10000)

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


def knn_expt_for_varied_tss():
    training_data_percentage = np.array([1,10,20,30,40,50,60,70,80,90,100]) * 0.01
    k = 1 #Found the value of k=1 by doing 10-fold cv

    rand_sample_for_tss_obj = mnist.MNIST_Random_Sample_Stratified() # Getting the MNIST_Random_Sample_Stratified object

    for t in range(len(training_data_percentage)):
        for i in range(NFOLDS):
            mnist_train, mnist_test = rand_sample_for_tss_obj.get_percentage_of_train_data(training_data_percentage[t]) # Getting train, test split for ith fold

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


def main():
    k_fold_cross_validation_for_choosing_k_in_knn()

if __name__=="__main__":
    main()