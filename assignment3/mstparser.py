"""Contains classes that can be used to train a maximum spanning tree parser:

- ParseVectorizer, which converts parsed sentences into feature vectors,

- MSTParsingDefinition, which contains the functions needed to train a parser
  with a learning algorithm such as the structured perceptron.

This module also includes a function to read NLTK's dependency treebank.
"""

from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import dependency_treebank
import math, itertools, numpy
from sklearn.pipeline import Pipeline
import sequence_tagger, conll_format
from sklearn.linear_model import Perceptron

def read_dependency_treebank():
    """Read the dependency treebank included in NLTK, and convert it to a
    simplified format.
    
    A sentence x is represented as a list of word/part-of-speech pairs, the
    first of which is a special dummy token '<TOP>' which will be the root of
    every parse tree. A parse tree y as a list of integers corresponding to the
    positions of the heads of the respective tokens; the first integer in this
    list is always -1, meaning that the dummy token has no head.

    For instance, if we have the sentence "John sleeps.", it will be represented
    as the list 

        [('<TOP>', '<TOP>'), ('John', 'NNP'), ('sleeps', 'VBZ'), ('.', '.')]

    and its parse tree will be

        [-1, 2, 0, 2]

    """
    XY = (convert_dependency_tree(t) for t in dependency_treebank.parsed_sents())
    X, Y = (list(t) for t in zip(*XY))
    return X, Y

class ParseVectorizer(object):
    """This class converts sentence/tree pairs into feature vectors.
    
    It has fit and transform methods similar to scikit-learn's vectorizer
    classes.
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, Y):
        """Create the mapping from parse tree features to vector space 
        dimensions."""
        seen_pairs = self.collect_pairs(X, Y)
        self.pfe = ParseFeatureExtractor(seen_pairs)
        self.dv = DictVectorizer()
        all_edges = []
        for x, y in zip(X, Y):
            for h, d in zip(y, range(len(y))):
                all_edges.append(self.pfe.extract_features(x, h, d))        
        all_edges.append({'unseen':True})
        self.dv.fit(all_edges)

    def transform(self, X, Y):
        """
        Extract features from the sentences and parse trees, and convert them
        into sparse matrices.

        Returns two lists: the transformed sentences and the transformed trees.

        To call this method, you must have called the fit method previously.
        """        
        print("Transforming all edges...")
        all_edge_features = self.dv.transform(self.pfe.extract_features(x, h, d) 
                                              for x in X 
                                              for (h,d) in itertools.product(range(len(x)), range(len(x))))
        lengths = (len(x)*len(x) for x in X)
        return [ all_edge_features[i:j] for (i,j) in self.starts_ends(lengths) ], Y
    
    def number_of_features(self):
        """Return the number of features used by this vectorizer."""
        return len(self.dv.feature_names_)
    
    def fit_transform(self, X, Y):
        """Equivalent to calling fit and then transform."""
        self.fit(X, Y)
        return self.transform(X, Y)

    ### internal helper methods

    def collect_pairs(self, X, Y):
        seen_pairs = set()
        for x, y in zip(X, Y):
            for d in range(1, len(y)):
                h = y[d]
                seen_pairs.add((x[h][1], x[d][1]))
        return seen_pairs

    def starts_ends(self, lengths):
        start = 0
        for l in lengths:
            yield (start, start + l)
            start += l


class MSTParsingDefinition(object):
    """This is the "problem definition" for dependency parsing. 
    
    It contains the  functions needed to train a parser with a learning 
    algorithm such as the structured perceptron:

     - finding the top-scoring parse tree, given a weight vector and a
       sentence,
       
     - getting the features for a parsed sentence.
     
    """
    
    def __init__(self, vec):
        pass
    
    def predict(self, w, X):
        """Given a weight vector w and unparsed input, return the top-scoring
        output.
        
        This method is the implementation for dependency parsing of the line

            guess = argmax_y  w * f(x, y) 

        in the structured perceptron pseudocode. The Eisner algorithm is used
        to find the top-scoring tree, so the method will always return a
        projective dependency tree.

        The method can be called with a single sentence or a list of sentences.
        """
        if isinstance(X, list):
            return [ eisner_search(w, x) for x in X ]
        else:
            return eisner_search(w, X)

    def get_features(self, x, y):
        """Return the features for the sentence/tree pair (x, y).
        
        The output is a sparse matrix with one row for each edge in the tree.
        """
        return x[vector_indices(y)]
    
    def print_info(self):
        """Print some time measurement information.
        
        Can be used e.g. after each perceptron iteration.
        """
        global scoring_time, eisner_time
        print("Total time scoring edges: {0}".format(scoring_time))
        print("Total time in Eisner search: {0}".format(eisner_time))
        scoring_time = 0
        eisner_time = 0


class ParseFeatureExtractor(object):
    """Feature extractor more or less like in McDonald's paper."""

    def __init__(self, seen_pairs):
        self.seen_pairs = seen_pairs
    
    def extract_features(self, x, head, dep):
        if head == dep or dep == 0:
            return {}

        p = x[head]
        c = x[dep]

        lr = "L" if dep < head else "R"
        
        features = {}

        # Basic unigram features
        features['bu1'] = p[0] + "/" + p[1] + "/" + lr
        features['bu2'] = p[0] + "/" + lr
        features['bu3'] = p[1] + "/" + lr
        features['bu4'] = c[0] + "/" + c[1] + "/" + lr
        features['bu5'] = c[0] + "/" + lr
        features['bu6'] = c[1] + "/" + lr
        
        # Basic bigram features
        features['bb1'] = p[0] + "/" + p[1] + "/" + c[0] + "/" + c[1] + "/" + lr
        features['bb2'] = p[1] + "/" + c[0] + "/" + c[1] + "/" + lr
        features['bb3'] = p[0] + "/" + c[0] + "/" + c[1] + "/" + lr
        features['bb4'] = p[0] + "/" + p[1] + "/" + c[1] + "/" + lr
        features['bb5'] = p[0] + "/" + p[1] + "/" + c[0] + "/" + lr
        features['bb6'] = p[0] + "/" + c[0] + "/" + lr
        features['bb7'] = p[1] + "/" + c[1] + "/" + lr
        
        # In-between POS features
        pcpos = p[1] + "/" + c[1] + "/" + lr + "/"
        for between in range(min(head, dep)+1, max(head, dep)):
            b = x[between]
            features['b=' + pcpos + b[1]] = True

        # Surrounding word POS features
        before_head_pos = "<OUT>" if head == 0 else x[head-1][1]
        after_head_pos = "<OUT>" if head == len(x)-1 else x[head+1][1]
        before_dep_pos = x[dep-1][1]
        after_dep_pos = "<OUT>" if dep == len(x)-1 else x[dep+1][1]

        features['sf1'] = p[1] + "/"+ after_head_pos + "/" + before_dep_pos + "/" + c[1] + "/" + lr
        features['sf2'] = before_head_pos + "/"+ p[1] + "/" + before_dep_pos + "/" + c[1] + "/" + lr
        features['sf3'] = p[1] + "/"+ after_head_pos + "/" + c[1] + "/" + after_dep_pos + "/" + lr
        features['sf4'] = before_head_pos + "/"+ p[1] + "/" + c[1] + "/" + after_dep_pos + "/" + lr
        
        # McDonald used a distance part in each feature. To save memory, we use 
        # it in just a few of them.
        dist = distance_bin(head, dep)
        features['dist'] = dist + "/" + lr
        features['dist_pp'] = dist + "/" + p[1] + "/" + lr
        features['dist_cp'] = dist + "/" + c[1] + "/" + lr
        features['dist_ppcp'] = dist + "/" + p[1] + "/" + c[1] + "/" + lr
        
        # A feature that says whether the POS pair was seen in the training set.
        # This is useful because we only use the features occurring in training
        # trees.        
        if (p[1], c[1]) not in self.seen_pairs:
            features['unseen'] = True
        
        return features

def distance_bin(h, d):
    dist = abs(h-d)
    if dist < 4:
        return str(dist)
    if dist < 8:
        return '4-7'
    if dist < 16:
        return '8-15'
    return 'long'

def vector_indices(y):
    n = len(y)
    return [ y[i]*n+i for i in range(1, n) ]

LEFT = 0
RIGHT = 1

import sys
import time

scoring_time = 0
eisner_time = 0

def collect_closed(cs, heads):
    upper, lower, _ = cs
    if upper:
        collect_open(upper, heads)
        collect_closed(lower, heads)

def collect_open(os, heads):
    head, dep, left, right, _ = os
    heads[dep] = head
    collect_closed(left, heads)
    collect_closed(right, heads)
 
def eisner_search(w, x):
    """The Eisner search algorithm, as in Appendix B in McDonald's paper."""
    global scoring_time, eisner_time

    t0 = time.time()
    
    edge_scores = x.dot(w)

    t1 = time.time()
    scoring_time += t1 - t0

    n_edges = len(edge_scores)
    n = int(math.sqrt(n_edges))
    
    LO = [[None for _ in range(n)] for _ in range(n)]
    LC = [[None for _ in range(n)] for _ in range(n)]
    RO = [[None for _ in range(n)] for _ in range(n)]
    RC = [[None for _ in range(n)] for _ in range(n)]


    for i in range(n):
        LC[i][i] = (None, None, 0.0)
        RC[i][i] = (None, None, 0.0)
    
    # The ranges are a bit different since McDonald's pseudocode indexes the
    # arrays from 1 to n.
    for k in range(1, n):
        for s in range(0, n):
            t = s + k
            if t >= n:
                break

            score, r = max( (RC[s][r][2] + LC[r+1][t][2], r) for r in range(s, t) )
            RO[s][t] = (s, t, RC[s][r], LC[r+1][t], score + edge_scores[n*s+t])

            LO[s][t] = (t, s, RC[s][r], LC[r+1][t], score + edge_scores[n*t+s])

            score, r = max( (RO[s][r][4] + RC[r][t][2], r) for r in range(s+1, t+1) )
            RC[s][t] = (RO[s][r], RC[r][t], score)

            score, r = max( (LC[s][r][2] + LO[r][t][4], r) for r in range(s, t) )
            LC[s][t] = (LO[r][t], LC[s][r], score)
            
    heads = [-1]*n
    collect_closed(RC[0][n-1], heads)
    
    t2 = time.time()
    eisner_time += t2 - t1
    
    return heads

def convert_dependency_tree(t):
    x = [ (node['word'], node['tag']) if node['word'] else ('<TOP>','<TOP>') for node in t.nodelist ]
    y = [ node['head'] if node['word'] else -1 for node in t.nodelist ]
    return x, y

class StructuredPerceptron():

    def __init__(self, n_features, problem, n_iter=20):
        self.n_features = n_features
        self.n_iter = n_iter
        self.problem = problem
    
    def add_sparse_rows_to_dense(self, fv, w, scale):
        prev = 0
        for ip in fv.indptr:
            w[fv.indices[prev:ip]] += scale*fv.data[prev:ip]
            prev = ip

    def sparse_dense_dot(slef, x, w):
        return numpy.dot(w[x.indices], x.data)

    def fit(self, X, Y):
        Y = list(Y)        
        self.w = numpy.zeros(self.n_features)

        # Converting the instance matrix to a list makes it faster to 
        # process one instance at a time when we're using sparse vectors.
        X = list(X) 
        for i in range(self.n_iter):            
            for x, y in zip(X, Y):
                yg    = self.problem.predict(self.w, x)
                phi   = self.problem.get_features(x, y)
                phi_g = self.problem.get_features(x, yg)

                self.add_sparse_rows_to_dense(phi, self.w, 1.0) 
                self.add_sparse_rows_to_dense(phi_g, self.w, -1.0)

    def predict(self, X):
        return self.problem.predict(self.w, X)

def runStructuredPerceptron():
    X,Y = read_dependency_treebank()
    traning_size = len(X) * 8 / 10  
    X_train, Y_train = X[:traning_size], Y[:traning_size]
    X_test, Y_test = X[traning_size:], Y[traning_size:]
    vec = ParseVectorizer()
    X_train,Y_train = vec.fit_transform(X_train,Y_train)
    n_features = vec.number_of_features()
    problem = MSTParsingDefinition(vec)
    sp = StructuredPerceptron(n_features, problem, 10)
    sp.fit(X_train, Y_train)
    Y_g = sp.predict(vec.transform(X_test, [])[0])
    error_counter = 0
    count = 0.0
    for ygs, ys in zip(Y_g, Y_test): 
        for yg, y in zip(ygs[1:], ys[1:]):
            if y != yg:
                error_counter += 1
            count += 1
    print (1 - (error_counter / count)) * 100
    problem.print_info()

def train_classifier(X, Y, classifier):
    vec = DictVectorizer()
    Xe = vec.fit_transform(X) # sparse matrix
    classifier.fit(Xe, Y)
    return Pipeline([('vec', vec), ('cls', classifier)])

def runExtractGreedyFeatures():
    X, Y = conll_format.read_sentences('./resources/eng.train.iob',200)
    X_test, Y_test = conll_format.read_sentences('./resources/eng.test.iob',50)

    # print X[1]
    vec = sequence_tagger.SequenceVectorizer()
    X_train,Y_train = vec.fit_transform(X, Y)
    n_features = vec.number_of_features()
    problem =  sequence_tagger.SequenceTaggingDefinition(vec)
    sp = StructuredPerceptron(n_features, problem, 10)
    print len(X_train)
    print len(Y_train)
    sp.fit(X_train, Y_train)
    Y_g = sp.predict_gready(vec.transform(X_test, [])[0])
    error_counter = 0
    count = 0.0
    for ygs, ys in zip(Y_g, Y_test): 
        for yg, y in zip(vec.to_tags(ygs), ys):
            if y != yg:
                error_counter += 1
            count += 1
    print (1 - (error_counter / count)) * 100
    problem.print_info()

if __name__ == '__main__':
    # runStructuredPerceptron()
    runExtractGreedyFeatures()