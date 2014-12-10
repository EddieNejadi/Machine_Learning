
from __future__ import print_function

# BaseEstimator and ClassifierMixin are base classes for classifiers in
# scikit-learn.
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy
from random import randrange

class LinearClassifier(BaseEstimator, ClassifierMixin):
    """Base class for linear classifiers, i.e. classifiers with a linear
    prediction function."""

    def find_classes(self, Y):
        """Find the set of output classes in a training set.

        The attributes positive_class and negative_class will be assigned
        after calling this method. An exception is raised if the number of
        classes in the training set is not equal to 2."""

        classes = list(set(Y))
        if len(classes) != 2:
            raise Exception("this does not seem to be a 2-class problem")
        self.positive_class = classes[0]
        self.negative_class = classes[1]

    def predict(self, X):        
        """Apply the linear scoring function and outputs self.positive_class
        for instances where the scoring function is positive, otherwise
        self.negative_class."""
        
        # To speed up, we apply the scoring function to all the instances
        # at the same time.
        scores = X.dot(self.w)
        
        # Create the output array.
        # At the positions where the score is positive, this will contain
        # self.positive class, otherwise self.negative_class.
        out = numpy.select([scores>=0.0, scores<0.0], [self.positive_class, 
                                                       self.negative_class])
        return out


class DensePerceptron(LinearClassifier):
    """A standard perceptron implementation, see slide 28 in the lecture."""

    def __init__(self, n_iter=10):
        self.n_iter = n_iter

    def fit(self, X, Y):
        Y = list(Y)
        self.find_classes(Y)

        # The feature vectors returned by DictVectorizer/CountVectorizer
        # are sparse vectors of the type scipy.sparse.csc_matrix. We convert
        # them to dense vectors of the type numpy.array.
        X = X.toarray()

        # The shape attribute holds the dimensions of the feature matrix. 
        # This is a tuple where
        # X.shape[0] = number of instances, 
        # X.shape[1] = number of features
        n_features = X.shape[1]

        # Initialize the weight vector to all zero
        self.w = numpy.zeros( n_features )
        
        for i in range(self.n_iter):            

            for x, y in zip(X, Y):
                
                # Compute the linear scoring function
                score = self.w.dot(x)                

                # If a positive instance was misclassified...
                if score < 0 and y == self.positive_class:
                    # then add its features to the weight vector
                    self.w += x

                # on the other hand if a negative instance was misclassified...
                elif score >= 0 and y == self.negative_class:
                    # then subtract its features from the weight vector
                    self.w -= x

def sign(y, pos):
    if y == pos:
        return 1.0
    else:
        return -1.0

class DensePerceptron2(LinearClassifier):
    """Reformulation of the perceptron where we encode the output classes as
    +1 and -1, see slide 45 in the lecture."""

    def __init__(self, n_iter=10):
        self.n_iter = n_iter

    def fit(self, X, Y):
        Y = list(Y)
        self.find_classes(Y)

        Yn = [sign(y, self.positive_class) for y in Y]

        X = X.toarray()
        n_features = X.shape[1]
        self.w = numpy.zeros( n_features )
        
        for i in range(self.n_iter):            
            for x, y in zip(X, Yn):
                score = self.w.dot(x) * y
                if score <= 0:
                    self.w += y*x

# Two helper functions for processing sparse and dense vectors.
# I haven't been able to do this efficiently in a more "civilised" manner: 
# these functions rely on the internal details of scipy.sparse.csr_matrix.
def add_sparse_to_dense(x, w, xw):
    w[x.indices] += xw*x.data
def sparse_dense_dot(x, w):
    return numpy.dot(w[x.indices], x.data)

class SparsePerceptron(LinearClassifier):
    """A perceptron implementation using sparse feature vectors, see 
    slide 30 in the lecture."""

    def __init__(self, n_iter=10):
        self.n_iter = n_iter

    def fit(self, X, Y):
        Y = list(Y)
        self.find_classes(Y)
        
        n_features = X.shape[1]
        self.w = numpy.zeros( n_features )

        # Converting the instance matrix to a list makes it faster to 
        # process one instance at a time when we're using sparse vectors.
        X = list(X) 

        for i in range(self.n_iter):            
            for x, y in zip(X, Y):
                score = sparse_dense_dot(x, self.w)
                if score < 0 and y == self.positive_class:
                    add_sparse_to_dense(x, self.w, 1.0)
                elif score >= 0 and y == self.negative_class:
                    add_sparse_to_dense(x, self.w, -1.0)
        

class Pegasos_SVM(LinearClassifier):

    def __init__(self, lam= 0.01, training_t=10):
        self.training_t = training_t
        self.lam = lam

    def fit(self, X, Y):
        Y = list(Y)
        self.find_classes(Y)
            
        n_features = X.shape[1]
        X = X.toarray()
        Y = list(Y)
        self.w = numpy.zeros( n_features )
        Yn = [sign(y, self.positive_class) for y in Y]
        self.training_t *= X.shape[0]
        numpy_vec = numpy.array(self.w)
        a = 1.0
        v = a * numpy.linalg.norm(numpy_vec)
        z = zip (X, Yn)
        my_counter = 0
        for t in range(self.training_t):
            (x, y) = z[t % X.shape[0]]
            eta_t_y = (1 / self.lam * (t+1)) * y
            c = (1 - ( 1 / (t+1) ))
            xw = self.w.dot(x)
            if t % 10000 == 0: print (xw)
            score = xw * y
            if score < 1 :
                self.w = (a * c) * self.w + (eta_t_y * x)
                a = 1.0
            else:
                a *= c
        if a != 1.0 : self.w *= a


class Logistic_regression(LinearClassifier):

    def __init__(self, lam= 0.01, training_t=10):
        self.training_t = training_t
        self.lam = lam

    def fit(self, X, Y):
        Y = list(Y)
        self.find_classes(Y)
            
        n_features = X.shape[1]
        X = X.toarray()
        Y = list(Y)
        self.w = numpy.zeros( n_features )
        Yn = [sign(y, self.positive_class) for y in Y]
        self.training_t *= n_features
        numpy_vec = numpy.array(self.w)
        z = zip (X, Yn)
        my_counter = 0
        for t in range(self.training_t):
            (x, y) = z[t % X.shape[0]]
            xw = self.w.dot(x)
            score = xw * y
            c = (1 - ( 1 / (t+1) ))
            self.w = c * self.w + (subgradiant * x)