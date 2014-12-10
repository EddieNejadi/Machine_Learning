
from __future__ import print_function, division

from nltk.corpus import treebank
from nltk.tree import Tree, ParentedTree

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split

#import sklearn.metrics
from sklearn.metrics import f1_score

import funtag_features
import treebank_helper

def train_scikit_classifier(X, Y):

    # Remove the tracing information from the feature dicts.
    for x in X:
        x.pop('_filename', None)
        x.pop('_sentence_id', None)
        x.pop('_phrase_id', None)

    # A DictVectorizer maps a feature dict to a sparse vector,
    # e.g. vec.transform({'phrase':'NP'}) might give [0, 0, ..., 0, 1, 0, ... ]
    vec = DictVectorizer()

    # Convert all the feature dicts to vectors.
    # As usual, it's more efficient to handle all at once.
    Xe = vec.fit_transform(X)

    # Initialize the learning algorithm we will use.
    
    # Use a perceptron classifier.
    # classifier = Perceptron(n_iter=20)

    # Use a Naive Bayes classifier.
    # classifier = MultinomialNB()

    # Use a support vector classifier.
    classifier = LinearSVC()

    # Finally, we can train the classifier.
    classifier.fit(Xe, Y)

    # Return a pipeline consisting of the vectorizer followed
    # by the classifier.
    return Pipeline([('vec', vec), ('classifier', classifier)])


def find_examples_in_tree(tree, X, Y, extractor,fe , filename, scount, ncount):
    """Apply the feature extractor to each phrase in a tree."""

    # we are in a terminal node -- this shouldn't happen
    if not isinstance(tree, Tree):
        raise Exception("this function shouldn't reach a terminal node")

    # we are in a preterminal node -- will not be handled
    if not isinstance(tree[0], Tree):
        return ncount + 2

    # call the user-provided feature extraction function
    features = extractor(tree,fe)

    # add tracing features so that we easily can print the example
    features['_filename'] = filename
    features['_sentence_id'] = scount
    features['_phrase_id'] = ncount

    # add the feature and correct function tags to the training set
    X.append(features)
    
    # some phrases have no function tag, or more than one
    if len(tree.funtags) == 0:
        Y.append('(empty)')
    else:
        Y.append('/'.join(sorted(tree.funtags)))

    # proceed recursively to the children of this phrase
    ncount += 1
    for subtree in tree:
        ncount = find_examples_in_tree(subtree, X, Y, extractor, 
                                   fe, filename, scount, ncount)
    return ncount

def read_treebank_files(files, extractor,fe):
    """Read the listed treebank files and collect function tagging examples
    from each tree.

    The user-provided feature extractor is applied to each phrase in each 
    tree. The extracted feature dicts and the true function tags for each
    phrase are stored in two separate lists, which are returned.
    """
    X = []
    Y = []
    for filename in files:
        scount = 0
        for tree in treebank.parsed_sents(filename):
            tree = ParentedTree.convert(tree)
            treebank_helper.postprocess(tree)
            find_examples_in_tree(tree, X, Y, extractor,fe, filename, scount, 0)
            scount += 1
    return X, Y

def write_example_tree(features, f):
    filename = features['_filename']
    sen = features['_sentence_id']
    phr = features['_phrase_id']
    tree = treebank.parsed_sents(filename)[sen]
    phrase = tree[tree.treepositions('preorder')[phr]]
    l = treebank_helper.get_label(phrase)
    treebank_helper.set_label(phrase, '***' + l + '***')
    f.write(str(tree))
    f.write('\n')
    treebank_helper.set_label(phrase, l)

def to_taglist(t):
    if t == '(empty)':
        return []
    else:
        return t.split('/')

def compute_statistics(Y_gold, Y_guess, stats):
    for y_gold, y_guess in zip(Y_gold, Y_guess):
        y_gold_tags = to_taglist(y_gold)
        y_guess_tags = to_taglist(y_guess)
        for t in y_gold_tags:
            stats[t].ntrue += 1
            stats['*ALL*'].ntrue += 1
            if t in y_guess_tags:
                stats[t].ncorrect += 1
                stats['*ALL*'].ncorrect += 1
        for t in y_guess_tags:
            stats[t].nguess += 1
            stats['*ALL*'].nguess += 1            

class PRF(object):
    def __init__(self):
        self.ncorrect = 0
        self.nguess = 0
        self.ntrue = 0

    def p(self):
        if self.nguess == 0:
            return 0.0
        return self.ncorrect/self.nguess

    def r(self):
        if self.ntrue == 0:
            return 0.0
        return float(self.ncorrect)/self.ntrue

    def f1(self):
        p = self.p()
        r = self.r()
        if p + r > 0:
            return 2*p*r/(p+r)
        else:
            return 0.0

    def all(self):
        return (self.ncorrect, self.nguess, self.ntrue, 
                self.p(), self.r(), self.f1())

def print_errors(X, tag, Ytrue, Yguess, logfile):
    with open(logfile, 'w') as f:
        f.write('*** False negatives: ***\n')
        for x, ytrue, yguess in zip(X, Ytrue, Yguess):        
            ytrue_tags = to_taglist(ytrue)
            yguess_tags = to_taglist(yguess)
            if tag in ytrue_tags and tag not in yguess_tags:
                f.write('Features: ' + str(x) + '\n')
                f.write('True tags: ' + str(ytrue_tags) + '\n')
                f.write('Guessed tags: ' + str(yguess_tags) + '\n')
                f.write('Tree:\n')
                write_example_tree(x, f)    
                f.write('\n')

        f.write('*** False positives: ***\n')
        for x, ytrue, yguess in zip(X, Ytrue, Yguess):        
            ytrue_tags = to_taglist(ytrue)
            yguess_tags = to_taglist(yguess)
            if tag not in ytrue_tags and tag in yguess_tags:
                f.write('Features:' + str(x) + '\n')
                f.write('True tags: ' + str(ytrue_tags) + '\n')
                f.write('Guessed tags: ' + str(yguess_tags) + '\n')
                f.write('Tree:\n')
                write_example_tree(x, f)
                f.write('\n')

def print_stats(Y_test, Y_out):
    all_tags = sorted(set(t for ts in (Y_test + list(Y_out))
                          for t in to_taglist(ts)))
    stats = {'*ALL*': PRF()}
    for tag in all_tags:
        stats[tag] = PRF()
    compute_statistics(Y_test, Y_out, stats)

    print('Overall statistics:')
    print('     #corr #guess  #true precision  recall  F-score')
    print('      {0: >4}   {1: >4}   {2: >4}   {3:0.4f}   {4:0.4f}   {5:0.4f}'.format(*stats['*ALL*'].all()))
    print('Statistics for each function tag:')
    print('tag  #corr #guess  #true precision  recall  F-score')
    for label in all_tags:
        print('{0}   {1: >4}   {2: >4}   {3: >4}   {4:0.4f}   {5:0.4f}   {6:0.4f}'.format(label, *stats[label].all()))


def lab1():
    # set this to your own feature extraction function

    extractor = funtag_features.extract_features
        
    # We reserve some treebank files for testing purposes.
    # This shouldn't be touched until you have optimized your
    # results on the development set.
    td_files, test_files = train_test_split(treebank.fileids(),
                                            train_size=0.8,
                                            random_state=0)

    # Split the rest into a training and a development set.
    train_files, dev_files = train_test_split(td_files,
                                              train_size=0.8,
                                              random_state=0)

    print('Reading training trees from treebank...')
    X_train, Y_train = read_treebank_files(train_files, extractor,[])

    print('Training classifier...')
    classifier = train_scikit_classifier(X_train, Y_train)

    print('Done training.')

    print('Reading evaluation trees from treebank...')
    X_eval, Y_eval = read_treebank_files(dev_files, extractor,[])

    # When you have optimized your system for the development set,
    # you can evaluate on the test set.
    #X_eval, Y_eval = read_treebank_files(test_files, extractor, [])

    print('Running classifier on evaluation data...')

    Y_out = classifier.predict(X_eval)

    print_stats(Y_eval, Y_out)

    # Uncomment this if you need detailed information about errors.
    # For instance, here is how you print false positives and negatives
    # of the LOC function to the file LOC.log.
    #print_errors(X_eval, 'LOC', Y_eval, Y_out, 'LOC.log')


def greedy_forward_selection():
    extractor = funtag_features.extract_features
    # funtag_features.availbale_features = []

    best_features = []
    best_f_score = 0.0
    all_features = ['label','head_pos','head','yeild','alt_head','alt_pos', 'parent_labels']
        # , 'grandmother_label','sister_labels','sister_poss','sister_head']
    
    td_files, test_files = train_test_split(treebank.fileids(),
                                            train_size=0.8,
                                            random_state=0)
    train_files, dev_files = train_test_split(td_files,
                                              train_size=0.8,
                                              random_state=0)
    while True:
        f_score_feature = []
        for f in all_features:
            # funtag_features.availbale_features.append(f)
            if f not in best_features:
                best_features.append(f)

            X_train, Y_train = read_treebank_files(train_files, extractor, best_features)
            classifier = train_scikit_classifier(X_train, Y_train)
            X_eval, Y_eval = read_treebank_files(dev_files, extractor, best_features)
            Y_out = classifier.predict(X_eval)
            print('Done training and evaluating with features:', best_features)
            stats = {'*ALL*': PRF()}
            all_tags = sorted(set(t for ts in (Y_eval + list(Y_out))
                      for t in to_taglist(ts)))
            for tag in all_tags:
                stats[tag] = PRF()
            compute_statistics(Y_eval, Y_out, stats)
            f_score_feature.append((stats['*ALL*'].f1(),f))
            best_features.remove(f)
        if best_f_score < max(f_score_feature)[0] :
            best_f_score = max(f_score_feature)[0]
            best_features.append(max(f_score_feature)[1])
            all_features.remove(max(f_score_feature)[1])
        else:
            break
        if len(best_features) == len(all_features):
            break
    print('Done with greedy_forward_selection..... ')
    print('Features are: ', best_features)
    print('Overall statistics:')
    print('     #corr #guess  #true precision  recall  F-score')
    print('      {0: >4}   {1: >4}   {2: >4}   {3:0.4f}   {4:0.4f}   {5:0.4f}'.format(*stats['*ALL*'].all()))
    print('Statistics for each function tag:')
    print('tag  #corr #guess  #true precision  recall  F-score')
    for label in all_tags:
        print('{0}   {1: >4}   {2: >4}   {3: >4}   {4:0.4f}   {5:0.4f}   {6:0.4f}'.format(label, *stats[label].all()))


def greedy_backward_selection():
    extractor = funtag_features.extract_features
    # funtag_features.availbale_features = []

    best_f_score = 0.0
    all_features = ['label','head_pos','head','yeild','alt_head','alt_pos', 'parent_labels']
        # , 'grandmother_label','sister_labels','sister_poss','sister_head']
    best_features = list(all_features)
    
    td_files, test_files = train_test_split(treebank.fileids(),
                                            train_size=0.8,
                                            random_state=0)
    train_files, dev_files = train_test_split(td_files,
                                              train_size=0.8,
                                              random_state=0)
    while True:
        f_score_feature = []
        for f in all_features:
            # funtag_features.availbale_features.append(f)
            if f in best_features:
                best_features.remove(f)

            X_train, Y_train = read_treebank_files(train_files, extractor, best_features)
            classifier = train_scikit_classifier(X_train, Y_train)
            X_eval, Y_eval = read_treebank_files(dev_files, extractor, best_features)
            Y_out = classifier.predict(X_eval)
            print('Done training and evaluating with features:', best_features)
            stats = {'*ALL*': PRF()}
            all_tags = sorted(set(t for ts in (Y_eval + list(Y_out))
                      for t in to_taglist(ts)))
            for tag in all_tags:
                stats[tag] = PRF()
            compute_statistics(Y_eval, Y_out, stats)
            f_score_feature.append((stats['*ALL*'].f1(),f))
            best_features.append(f)
        if best_f_score < max(f_score_feature)[0] :
            best_f_score = max(f_score_feature)[0]
            best_features.remove(max(f_score_feature)[1])
            all_features.remove(max(f_score_feature)[1])
        else:
            break
        if len(all_features) == 0:
            break
    print('Done with greedy_forward_selection..... ')
    print('Features are: ', best_features)
    print('Overall statistics:')
    print('     #corr #guess  #true precision  recall  F-score')
    print('      {0: >4}   {1: >4}   {2: >4}   {3:0.4f}   {4:0.4f}   {5:0.4f}'.format(*stats['*ALL*'].all()))
    print('Statistics for each function tag:')
    print('tag  #corr #guess  #true precision  recall  F-score')
    for label in all_tags:
        print('{0}   {1: >4}   {2: >4}   {3: >4}   {4:0.4f}   {5:0.4f}   {6:0.4f}'.format(label, *stats[label].all()))

if __name__ == '__main__':
    # lab1()
    greedy_forward_selection()
    greedy_backward_selection()
