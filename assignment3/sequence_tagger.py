"""Contains classes that can be used to train a sequence tagger:

- SequenceVectorizer, which converts tagged sentences into feature vectors,

- SequenceTaggingDefinition, which contains the functions needed to train a
  tagger with a learning algorithm such as the structured perceptron.
"""

from sklearn.feature_extraction import DictVectorizer

import itertools
import time
import conll_format as conll
import numpy

class SequenceVectorizer(object):
    """This class converts tagged sentences into feature vectors.
    
    It has fit and transform methods similar to scikit-learn's vectorizer
    classes.
    """
    
    def __init__(self):
        pass

    def fit(self, X, Y):
        """Create the mapping from tagged sentence features to vector space 
        dimensions."""
        self.find_tags(Y)        
        self.dv = DictVectorizer()
        self.fe = SequenceFeatureExtractor()
        all_steps_features = []
        for x, y in zip(X, Y):
            prev_tag = BEFORE_TAG
            for i in range(len(y)):
                all_steps_features.append(self.fe.extract_emission_features(x, i, y[i]))
                all_steps_features.append(self.fe.extract_transition_features(x, i, prev_tag, y[i]))
                prev_tag = y[i]
            all_steps_features.append(self.fe.extract_transition_features(x, len(y), prev_tag, AFTER_TAG))
        self.dv.fit(all_steps_features)        

    def transform(self, X, Y):
        """
        Extract features from the sentences and tag sequences, and convert them
        into sparse matrices.

        Returns two lists: the transformed sentences and the transformed tag
        sequences.

        To call this method, you must have called the fit method previously.
        """
        all_steps_features = []
        print("Extracting features for all possible emissions and transitions...")
        for x in X:
            ntokens = len(x)
            for i in range(ntokens):
                for tag in self.tags:
                    all_steps_features.append(self.fe.extract_emission_features(x, i, tag))
            for tag in self.tags:
                all_steps_features.append(self.fe.extract_transition_features(x, i, BEFORE_TAG, tag))
            for i in range(1, ntokens):
                for prev_tag, tag in itertools.product(self.tags, self.tags):
                    all_steps_features.append(self.fe.extract_transition_features(x, i, prev_tag, tag))
            for tag in self.tags:
                all_steps_features.append(self.fe.extract_transition_features(x, i, tag, AFTER_TAG))              
        print("Transforming features...")                
        all_steps_vectors = self.dv.transform(all_steps_features)
        print("Grouping features...")
        ntags = len(self.tags)
        lengths = ( len(x)*ntags + (len(x)-1)*ntags*ntags + 2*ntags for x in X)
        Xe = [ all_steps_vectors[i:j] for (i,j) in self.starts_ends(lengths) ]
        Ye = [ [(self.tag_to_id[t] if t in self.tag_to_id else -1) for t in y] for y in Y ]
        return Xe, Ye

    def fit_transform(self, X, Y):
        """Equivalent to calling fit and then transform."""
        self.fit(X, Y)
        return self.transform(X, Y)

    def number_of_features(self):
        """Return the number of features used by this vectorizer."""
        return len(self.dv.feature_names_)

    def to_tags(self, y):
        """Convert an encoded tag sequence into a sequence of string tags."""
        return [ self.tags[i] for i in y ]

    # internal helper methods

    def find_tags(self, Y):
        ts = set()
        for y in Y:
            ts.update(y)
        self.tags = sorted(ts)
        self.tag_to_id = { self.tags[i]:i for i in range(len(self.tags)) }

    def starts_ends(self, lengths):
        start = 0
        for l in lengths:
            yield (start, start + l)
            start += l

class SequenceTaggingDefinition(object):
    """This is the "problem definition" for sequence tagging.
    
    It contains the  functions needed to train a tagger with a learning 
    algorithm such as the structured perceptron:

     - finding the top-scoring tag sequence, given a weight vector and a
       sentence,
       
     - getting the features for a tagged sentence.
     
    """
    def __init__(self, vec):
        self.ntags = len(vec.tags)

    def predict(self, w, X):
        """Given a weight vector w and untagged input, return the top-scoring
        output.
        
        This method is the implementation for sequence tagging of the line

            guess = argmax_y  w * f(x, y) 

        in the structured perceptron pseudocode. The Viterbi algorithm is used
        to find the top-scoring sequence.

        The method can be called with a single sentence or a list of sentences.
        """
        if isinstance(X, list):
            return [ viterbi(w, x, self.ntags) for x in X ]
        else:
            return viterbi(w, X, self.ntags)

    def get_features(self, x, y):
        """Return the features for the sentence/sequence pair (x, y).
        
        The output is a sparse matrix with multiple rows, depending on the
        number of tokens in the sentence."""        
        global get_features_time
        t0 = time.time()
        
        nt = self.ntags
        ntokens = len(y)
        emissions = [ nt*i+y[i] for i in range(ntokens) ]        
        first_transition = [ nt*ntokens + y[0] ]        
        transitions = [ nt*ntokens + nt + (i-1)*nt*nt + y[i-1]*nt + y[i] 
                        for i in range(1, ntokens) ]        
        last_transition = [ nt*ntokens + nt + nt*nt*(ntokens-1) + y[-1] ]        

        out = x[ emissions + first_transition + transitions + last_transition ]

        t1 = time.time()
        get_features_time += t1 - t0
        
        return out

    def print_info(self):
        """Print some time measurement information.
        
        Can be used e.g. after each perceptron iteration.
        """
        
        global scoring_time, viterbi_time, get_features_time
        print("Total time scoring emissions and transitions: {0}".format(scoring_time))
        print("Total time in Viterbi search: {0}".format(viterbi_time))
        print("Total time in get_features: {0}".format(get_features_time))
        scoring_time = 0
        viterbi_time = 0
        get_features_time = 0

scoring_time = 0
viterbi_time = 0
get_features_time = 0

class SequenceFeatureExtractor(object):
    """Feature extractor containing some typical named-entity recognition
    features."""

    def __init__(self):
        pass
    
    def extract_greedy_features(self, x, position, prev_tag):
        features = self.extract_emission_features(x, position, "?")
        features.update(self.extract_transition_features(x, position, prev_tag, "?"))
        return features
    
    def extract_emission_features(self, x, position, tag):
        features = {}
        features['w_i'] = x[position][0] + "/" + tag
        features['p_i'] = x[position][1] + "/" + tag
        features['c_i'] = x[position][2] + "/" + tag

        features['sf4_i'] = x[position][0][-4:] + "/" + tag
        features['sf3_i'] = x[position][0][-3:] + "/" + tag
        features['sf2_i'] = x[position][0][-2:] + "/" + tag
        features['sf1_i'] = x[position][0][-1:] + "/" + tag

        if position > 0:
            features['w_i-1'] = x[position-1][0] + "/" + tag
            features['p_i-1'] = x[position-1][1] + "/" + tag
        else:
            features['w_i-1'] = BEFORE_TAG + "/" + tag
            features['p_i-1'] = BEFORE_TAG + "/" + tag

        if position < len(x)-1:
            features['w_i+1'] = x[position+1][0] + "/" + tag
            features['p_i+1'] = x[position+1][1] + "/" + tag
        else:
            features['w_i+1'] = AFTER_TAG + "/" + tag
            features['p_i+1'] = AFTER_TAG + "/" + tag            
        
        return features
    
    def extract_transition_features(self, x, position, prev_tag, tag):
        features = {}
        features['transition'] = prev_tag + "/" + tag
        return features

BEFORE_TAG = "<BEFORE>"
AFTER_TAG = "<AFTER>"
NEG_INF = float("-inf")

def viterbi(w, x, ntags):
    """The Viterbi algorithm for finding the best tag sequence."""
    global scoring_time, viterbi_time

    t0 = time.time()
    scores = x.dot(w)
    t1 = time.time()
    scoring_time += t1 - t0

    nfv = len(scores)
    ntokens = int((nfv + ntags*ntags - 2*ntags)/(ntags + ntags*ntags))

    first_item = (None, None, 0.0)
    items = [ viterbi_first_step(scores, tag, ntags, 
                                 ntokens, first_item) 
              for tag in range(ntags) ]    
    for position in range(1, ntokens):
        prev_items = items
        items = [ viterbi_next_step(scores, position, tag, 
                                    ntags, ntokens, prev_items) 
                  for tag in range(ntags) ]
    finalitem = viterbi_last_step(scores, ntags, ntokens, items)
    
    out = [-1] * ntokens
    i = finalitem[1]
    position = ntokens
    while position > 0:
        position -= 1
        out[position] = i[0]
        i = i[1]
    
    t2 = time.time()
    viterbi_time += t2 - t1
    
    return out

def viterbi_first_step(scores, tag, ntags, ntokens, first_item):
    transition_score = scores[ntags*ntokens + tag]
    emission_score = scores[tag]
    return (tag, first_item, transition_score + first_item[2] + emission_score)

def viterbi_next_step(scores, position, tag, ntags, ntokens, prev_items):
    # we know position > 0 and position < ntokens
    offset = ntags*ntokens + ntags + (position-1)*ntags*ntags + tag
    # this is faster than maximizing over a generator for some reason...
    max_score = NEG_INF    
    for prev in prev_items:
        transition_score = scores[offset + prev[0]*ntags]
        score = transition_score + prev[2]
        if score > max_score:
            max_score = score
            max_prev = prev
    emission_score = scores[position*ntags + tag]
    return (tag, max_prev, max_score + emission_score)

def viterbi_last_step(scores, ntags, ntokens, prev_items):
    offset = ntags*ntokens + ntags + (ntokens-1)*ntags*ntags    
    max_score = NEG_INF
    for prev in prev_items:
        transition_score = scores[offset + prev[0]]
        score = transition_score + prev[2]
        if score > max_score:
            max_score = score
            max_prev = prev
    return (None, max_prev, max_score)


class StructuredPerceptron(object):

    "According to Collins (2002) optimal number of iteration is 20"
    def __init__(self, n_iter = 20):
        self.n_iter= n_iter

    def fit(self, X, Y):
        Y = list(Y)
        self.find_classes(Y)


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



if __name__ == '__main__':
    # Reading from the file
    X, Y = conll.read_sentences('./resources/eng.train.iob', 100)
    X_test, Y_test = conll.read_sentences('./resources/eng.test.iob', 50)


    vec = SequenceVectorizer()
    X_train,Y_train = vec.fit_transform(X, Y)
    n_features = vec.number_of_features()
    problem =  SequenceTaggingDefinition(vec)
    sp = StructuredPerceptron(n_features, problem, 10)
    sp.fit(X_train, Y_train)
    Y_g = sp.predict(vec.transform(X_test, [])[0])
    error_counter = 0
    count = 0.0
    for ygs, ys in zip(Y_g, Y_test): 
        for yg, y in zip(vec.to_tags(ygs), ys):
            if y != yg:
                error_counter += 1
            count += 1
    print (1 - (error_counter / count)) * 100
    problem.print_info()


