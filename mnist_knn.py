import mnist_data as mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import time
import numpy as np

mnist_all = mnist.get_MNIST_data()

for k in xrange(60):
    mnist_train_k, mnist_test_k = mnist.MNIST_train_test_split_k(mnist_all, (k + 1)*1000)

    print "Number of training samples: {}".format((k+1)*1000)
    print "Number of test samples:{}".format(len(mnist_test_k.data))

	# Nearest neighbor
    t0 = time.clock()
    knn = KNeighborsClassifier() #All default parameters
    knn.fit(mnist_train_k.data, mnist_train_k.target)
    execTime = time.clock() - t0

    print "Execution Time for Nearest Neighbor: {}".format(execTime)

    t0 = time.clock()
    y_pred = knn.predict(mnist_test_k.data)
    execTime = time.clock() - t0

    print "Execution Time for Nearest Neighbor Prediction: {}".format(execTime)

    print "Accuracy: {}".format(metrics.accuracy_score(mnist_test_k.target, y_pred))