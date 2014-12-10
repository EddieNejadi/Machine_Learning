
"""Functions for reading and writing the CoNLL column format, and for calling
the CoNLL evaluation script."""

import os

def run_evaluator(filename):
    """Run the official CoNLL evaluation script. 
    
    The last two columns of the given file must consist of true tags and
    predicted tags, respectively. Perl must be installed on the system.
    """
    os.system("perl conll2003/conlleval.perl < " + filename)

COLUMN_SEPARATOR = " "

def read_sentences(filename, max_n_sentences=-1):
    """Read tagged sentences from a text file. 

The sentences are formatted in a row-column format. One row 
contains a list of features for one word, and the last column  
contains the output tags. The columns are separated by the string 
COLUMN_SEPARATOR. The sentences are separated by empty lines.

Example:

Estimated VBN B-NP
volume NN I-NP
was VBD O
a DT B-NP
light NN I-NP
2.4 CD I-NP
million CD I-NP
ounces NNS I-NP
. . O

Return a two lists: a list of sentences, and a list of tag sequences for
each sentence. In the example above, we return

[[("Estimated", "VBN"), ("volume", "NN"), "I-NP"), ..., (".", "."), "O")], ...]

and 

[["B-NP", "I-NP", ..., "O"], ...]
    
"""
    with open(filename) as f:
        X = []
        Y = []
        x = []
        y = []
        l = f.readline()
        while True:
            if l == "":
                return X, Y
            l = l.strip()
            if l == "":
                X.append(x)
                Y.append(y)
                if len(Y) == max_n_sentences:
                    return X, Y
                x = []
                y = []
            else:
                tokens = l.split(COLUMN_SEPARATOR)
                x.append(tuple(tokens[:-1]))
                y.append(tokens[-1])
            l = f.readline()


def write_sentence(f, sentence, tags, tags2=None):
    """
Write a sentence, a list of tags, and an optional second list of tags to a text file. 

The sentence is a list of tuples, as returned by read_sentences. tags and tags2 
are lists of strings. The sentence is written in the same format as read
by read_sentences. tags and tags2 are written as the columns after the
sentence columns.
    """
    
    for i in range(len(sentence)):
        for t in sentence[i]:
            f.write(t)
            f.write(COLUMN_SEPARATOR)
        f.write(tags[i])
        if tags2:
            f.write(COLUMN_SEPARATOR)
            f.write(tags2[i])
        f.write("\n")
    f.write("\n")
