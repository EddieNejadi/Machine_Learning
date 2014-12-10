
from treebank_helper import get_label, get_parent

# Comment:
# get_label and get_parent are helper functions to handle compatibility
# issues between NLTK versions 2 and 3.
#
# NLTK 2: 
#  get_label(tree) is tree.node
#  get_parent(tree) is tree.parent
# NLTK 3:
#  get_label(tree) is tree.label()
#  get_parent(tree) is tree.parent()

def extract_features(tree,fe=[]):
    if len(fe) == 0: 
        fe = ['label','head_pos','head','yeild','alt_head','alt_pos', 'parent_labels', 'grandmother_label','sister_labels','sister_poss','sister_head']
    features = {}

    if 'label' in fe:
        features['label'] = get_label(tree)
    
    # INSERT YOUR FEATURE EXTRACTION CODE HERE

    # some useful things you may need when extracting features

    # get_label(tree): the phrase label
    # get_parent(tree): the parent phrase (or None if we are at the top)
    # tree.head: the head token of the phrase (this is also a small tree)

    # also, each tree behaves like a list of its children, so we can do
    # len(tree): the number of children
    # tree[n]: the n'th child

    if 'head_pos' in fe:
        features['head_pos'] = get_label(tree.head)
    if 'head' in fe:
        features['head'] = tree.head[0]
    if 'yeild' in fe:
        features['yeild'] = len(tree.flatten())

    if get_label(tree) == 'PP' and len(tree) == 2 and len(tree[1]) == 1:
        if 'alt_head' in fe:
            features['alt_head'] = tree[1].head[0]
        if 'alt_pos' in fe:
            features['alt_pos'] = get_label(tree[1].head)

    parent = get_parent(tree)
    if parent:
        if 'parent_labels' in fe:
            features['parent_labels'] = get_label(parent)
        grandmother = get_parent(parent)
        if 'grandmother_label' in fe and grandmother:
            features['grandmother_label'] = get_label(grandmother)

        parent.remove(tree)
        parent_head = parent.head
        if len(parent) > 0 :
            if 'sister_labels' in fe:
                features['sister_labels'] = get_label(parent[0])
            if 'sister_poss' in fe:
                features['sister_poss'] = get_label(parent[0].head)
            if 'sister_poss' in fe:
                features['sister_head'] = parent[0].head[0]

    return features
