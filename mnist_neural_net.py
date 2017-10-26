import mnist_data as mnist
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import time

mnist_all = mnist.get_MNIST_data()

mnist_train = list()
mnist_test = list()

for k in xrange(60):  
    mnist_train_k, mnist_test_k = mnist.MNIST_train_test_split_k(mnist_all, (k + 1)*1000)
    mnist_train.append(mnist_train_k)
    mnist_test.append(mnist_test_k)

    print "Number of training samples: {}".format((k+1)*1000)

	# Multi-Layer Perceptron
    t0 = time.clock()
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                        solver='sgd', tol=1e-4, random_state=1,
                        learning_rate_init=.1)
    mlp.fit(mnist_train_k.data, mnist_train_k.target)
    execTime = time.clock() - t0
                
    print "Execution Time for Neural Net: {}".format(execTime)                         
    
    y_pred = mlp.predict(mnist_test_k.data)
    print "Accuracy: {}".format(metrics.accuracy_score(mnist_test_k.target, y_pred))