
from __future__ import print_function

import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

import ml_lecture3

def read_corpus(corpus_file):
    X, Y = [], []
    with open(corpus_file) as f:
        for line in f:
            tokens = line.strip().split()
            Y.append(tokens[1])
            X.append(tokens[3:])
    return X, Y

def train_classifier(X, Y, classifier):
    vec = CountVectorizer(preprocessor=lambda x: x,
                          tokenizer=lambda x: x,
                          binary=True)
    Xe = vec.fit_transform(X)
    classifier.fit(Xe, Y)
    return Pipeline([('vec', vec), ('cls', classifier)])

def assignment2_experiment():
    X_all, Y_all = read_corpus("all_sentiment_shuffled.txt")
    X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all,
                                                        train_size=0.8,
                                                        random_state=0)

    t1 = time.time()
        
    # using one of the perceptron implementations from the lecture
    # classifier = train_classifier(X_train, Y_train, 
                                  # ml_lecture3.SparsePerceptron(n_iter=5))

    # using classifiers from scikit-learn
    # classifier = train_classifier(X_train, Y_train, Perceptron(n_iter=5))
    # classifier = train_classifier(X_train, Y_train, LinearSVC())
    # classifier = train_classifier(
    #     X_train, Y_train, ml_lecture3.Pegasos_SVM(0.01,10))
    classifier = train_classifier(
        X_train, Y_train, ml_lecture3.Logistic_regression(0.01,10))

    t2 = time.time()

    print('Training time: {0:.3f} sec.'.format(t2-t1))

    t2c = time.time()

    Y_guesses = classifier.predict(X_test)

    t2d = time.time()
    print('Classification time: {0:.3f} sec.'.format(t2d-t2c))

    acc = accuracy_score(Y_test, Y_guesses)
    print('Accuracy on the test set: {0:.3f}'.format(acc))





if __name__ == '__main__':
    assignment2_experiment()

