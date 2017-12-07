from collections import namedtuple
from sklearn.datasets import fetch_mldata
import numpy as np
from sys import platform
from sklearn.decomposition import PCA

MNIST_data = namedtuple("MNIST_data", "data, target")
NUMBER_OF_CLASSES = 10


def get_MNIST_data():
    if platform == "darwin": # Mac OS
        data_dir = "../../MNIST_Data"
    elif platform == "win32": # Windows
        data_dir = "D:\\Vinoth\\Course_work\\CS 578\\Project\\MNIST_Data"
    else:
        data_dir = "../MNIST_Data"

    mnist = fetch_mldata('MNIST original', data_home = data_dir)
    return mnist


def MNIST_train_test_split():
    mnist = get_MNIST_data()

    mnist_train = MNIST_data(mnist.data[:60000,:], mnist.target[:60000])
    mnist_test = MNIST_data(mnist.data[60000:,:], mnist.target[60000:])

    return mnist_train, mnist_test


# k-fold without stratification
def MNIST_train_test_split_k_fold(mnist, number_of_fold, fold_no):
    """
    :param mnist: data.
    :param number_of_fold: Number of folds e.g 10 fold cv.
    :param fold_no: Which fold it is
    :return: train, test data.

    """
    indices = np.arange(len(mnist.data))
    set_size = len(mnist.data) / number_of_fold

    print 'set_size is:',set_size

    if fold_no == 0:
        np.random.shuffle(indices)

    test_set_indices = indices[set_size * fold_no : (set_size * (fold_no + 1))]
    training_set_indices = np.hstack((indices[0:(set_size * fold_no)],
                                      indices[(set_size * (fold_no + 1)):]))

    mnist_train = MNIST_data(mnist.data[training_set_indices,:], mnist.target[training_set_indices])
    mnist_test = MNIST_data(mnist.data[test_set_indices], mnist.target[test_set_indices])

    return mnist_train, mnist_test


# k-fold CV with stratification
class MNIST_CV_Stratified:

    def __init__(self, mnist_data):
        self.mnist_data = mnist_data
        self.indices_per_class = [None] * NUMBER_OF_CLASSES;
        number_of_data_per_class = [None] * NUMBER_OF_CLASSES;

        for i in range(NUMBER_OF_CLASSES):
            self.indices_per_class[i] = np.argwhere(mnist_data.target[:] == i).flatten()
            number_of_data_per_class[i] = len(self.indices_per_class[i])

        number_of_data_per_class = np.array(number_of_data_per_class)
        print 'percent data per class :', (number_of_data_per_class / float(len(mnist_data.target))) * 100


    def get_train_test_split(self, number_of_fold, fold_no):
        """
        :param mnist_data: data.
        :param number_of_fold: Number of folds e.g 10 fold cv.
        :param fold_no: Which fold it is
        :return: train, test data.

        """

        indices = np.arange(len(self.mnist_data.data))

        test_set_indices = []

        for i in range(NUMBER_OF_CLASSES):
            set_size_per_class = len(self.indices_per_class[i]) / number_of_fold

            if fold_no == number_of_fold - 1:
                test_set_indices.extend(self.indices_per_class[i][set_size_per_class * fold_no : ])

            else:
                test_set_indices.extend(
                        self.indices_per_class[i][set_size_per_class * fold_no : (set_size_per_class * (fold_no + 1))])

        training_set_indices = np.setdiff1d(indices,test_set_indices)

        mnist_train = MNIST_data(self.mnist_data.data[training_set_indices, :], self.mnist_data.target[training_set_indices])
        mnist_test = MNIST_data(self.mnist_data.data[test_set_indices], self.mnist_data.target[test_set_indices])

        #print "==== Fold: %d ====" % fold_no
        #print "train labels:", mnist_train.target
        #print "test labels:", mnist_test.target
        #print "train indices:", training_set_indices
        #print "test indices:", test_set_indices

        return mnist_train, mnist_test


# Randomly sample data with stratification
class MNIST_Random_Sample_Stratified:

    def __init__(self):
        self.mnist_train, self.mnist_test =  MNIST_train_test_split()

        self.indices_per_class_in_train = [None] * NUMBER_OF_CLASSES;
        number_of_data_per_class_in_train = [None] * NUMBER_OF_CLASSES;

        for i in range(NUMBER_OF_CLASSES):
            self.indices_per_class_in_train[i] = np.argwhere(self.mnist_train.target[:] == i).flatten()
            number_of_data_per_class_in_train[i] = len(self.indices_per_class_in_train[i])

        number_of_data_per_class_in_train = np.array(number_of_data_per_class_in_train)
        print 'percent data per class in train:', \
            (number_of_data_per_class_in_train / float(len(self.mnist_train.target))) * 100


    def sample_train_data(self, training_data_percentage):
        """
        :param mnist_data: data.
        :param training_data_percentage: Training data percentage for varied tss experiment
        :return: train, test data. Test data is always 60000-70000. Keeping test dataset constant

        """

        training_set_indices = []

        for i in range(NUMBER_OF_CLASSES):
            num_of_data_per_class = int(len(self.indices_per_class_in_train[i]) * training_data_percentage)
            # Randomly sampling num_of_data_per_class data from each class
            indices_i = np.random.choice(self.indices_per_class_in_train[i], size=num_of_data_per_class, replace=False)
            training_set_indices.extend(indices_i)

        mnist_train_sampled = MNIST_data(self.mnist_train.data[training_set_indices, :],
                                         self.mnist_train.target[training_set_indices])

        #print "==== Percentage: %f ====" % training_data_percentage
        #print "train labels:", mnist_train_sampled.target
        #print "train indices:", training_set_indices

        return mnist_train_sampled, self.mnist_test


def MNIST_pca(mnist):
    # Mean centering
    mnist_data_centered = mnist.data - mnist.data.mean(axis=0)
    pca = PCA()
    pca.fit(mnist_data_centered)
    var_explained = pca.explained_variance_ratio_
    n_components = np.min(np.where (np.cumsum(var_explained) > 0.9)) + 1
    pc = pca.components_[:n_components, :].T
    return pc, mnist.data.mean(axis=0)


def PCA_transform(mnist, pc, mean):
    mnist_data_centered = mnist.data - mean
    transformed_data = mnist_data_centered.dot(pc)
    mnist_transform = MNIST_data(transformed_data, mnist.target)
    return mnist_transform


def main():
    #nfolds = 3
    #mnist_data = get_MNIST_data()

    #data = np.random.randn(20, 2)
    #target = np.array([0, 0, 1, 1, 1, 2, 2, 2, 0, 1] * 2)
    mnist_data = MNIST_data(data, target)

    #cv_obj = MNIST_CV_Stratified(mnist_data)

    #for i in range(nfolds):
     #   cv_obj.get_train_test_split(nfolds, i)

    #sample_obj = MNIST_Random_Sample_Stratified(mnist_data)
    #sample_obj.sample_train_data(0.5)
    pass


if __name__=="__main__":
    main()


