import mnist_utils as mnist
import numpy as np

mnist_all = mnist.get_MNIST_data()
knn_training_data_class_count = np.zeros((10,10), dtype=float)
knn_test_data_class_count = np.zeros((10,10), dtype=float)

NFOLDS = 10

for i in range(NFOLDS):
    for j in range(10):

        mnist_train, mnist_test = mnist.MNIST_train_test_split_k_fold(mnist_all, NFOLDS, i)

        print "Number of training samples: {}".format(len(mnist_train.data))
        print "Number of test samples:{}".format(len(mnist_test.data))

        knn_training_data_class_count[i][j] = len(np.argwhere(mnist_train.target[:] == j).flatten())

        knn_test_data_class_count[i][j] = len(np.argwhere(mnist_test.target[:] == j).flatten())

np.savetxt("knn_training_data_class_counts.csv", knn_training_data_class_count, delimiter=",", fmt="%.3f")
np.savetxt("knn_test_data_class_counts.csv", knn_test_data_class_count, delimiter=",", fmt="%.3f")