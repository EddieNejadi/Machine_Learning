"""Includes functions to process the trees for ease of use in the lab.

First splits off the function tags from the phrase labels. Then applies
a procedure to determine the syntactic heads of the phrases. The head
extraction rules are taken from Collins' PhD thesis, with some small
modifications.
"""

from __future__ import print_function, division

import re

import nltk
from nltk.corpus import treebank
from nltk.tree import Tree, ParentedTree

# tree processing differs between NLTK 2 and 3
if nltk.__version__[0] == '3':
    def get_label(c):
        return c.label() or c.node    
    def set_label(c, l):
        c.set_label(l)
    def get_parent(c):
        return c.parent()
else:
    def get_label(c):
        return c.node    
    def set_label(c, l):
        c.node = l
    def get_parent(c):
        return c.parent()

def postprocess(tree):
    find_function_tags(tree)
    set_heads(tree)

def find_function_tags(tree):
    """Determines the function tags for each phrase in a Penn Treebank tree.

    After calling this function, each phrase has an attribute called funtags
    containing a list of function tags for that phrase. Function tags and 
    trace ids are removed from the phrase labels.

    For instance, if the phrase p was originally tagged NP-SBJ-2, after calling
    this function p.label() will return "NP" and p.funtags will be ["SBJ"].
    """
    if not isinstance(tree, Tree):
        return

    l = get_label(tree).strip()

    if l == '-NONE-':
        if tree[0][0] == '*':
            tree[0] = '*'
        tree.funtags = []
        return
    elif re.match('(-[LR][CSR]B-)', l):
        tree.funtags = []        
        return
    ls = l.split('-')

    set_label(tree, remove_gaplabel(ls[0]))

    tree.funtags = [remove_gaplabel(t) for t in ls[1:] if not is_number(t)]
    for subtree in tree:
        find_function_tags(subtree)

def is_number(s):
    return re.match('^[0-9]+$', s)

def remove_gaplabel(s):
    index = s.find('=')
    if index > 0:
        return s[:index]
    return s


RULE_TABLE = {'ADJP': ('l', ['NNS', 'QP', 'NN', '$', 'ADVP', 'JJ', 'VBN',
                             'VBG', 'ADJP', 'JJR', 'NML', 'NP', 'JJS', 'DT',
                             'FW', 'RBR', 'RBS', 'SBAR', 'RB']),
              'ADVP': ('r', ['RB', 'RBR', 'RBS', 'FW', 'ADVP', 'TO', 'CD',
                             'JJR', 'JJ', 'IN', 'NML', 'NP', 'JJS', 'NN']),
              'CONJP': ('r', ['CC', 'RB', 'IN']),
              'FRAG': ('r', []),
              'INTJ': ('l', []),
              'LST': ('r', ['LS', ':']),
              'PP': ('r', ['IN', 'TO', 'VBG', 'VBN', 'RP', 'FW']),
              'PRN': ('l', []),
              'PRT': ('r', ['RP']),
              'QP': ('l', ['$', 'IN', 'NNS', 'NN', 'JJ', 'RB', 'DT', 'CD',
                           'NCD', 'QP', 'JJR', 'JJS']),
              'RRC': ('r', ['VP', 'NP', 'ADVP', 'ADJP', 'PP']),
              'S': ('l', ['TO', 'IN', 'VP', 'S', 'SBAR', 'ADJP', 'UCP', 'NP']),
              'SBAR': ('l', ['WHNP', 'WHPP', 'WHADVP', 'WHADJP', 'IN', 'DT',
                             'S', 'SQ', 'SINV', 'SBAR', 'FRAG']),
              'SBARQ': ('l', ['SQ', 'S', 'SINV', 'SBARQ', 'FRAG']),
              'SINV': ('l', ['VBZ', 'VBD', 'VBP', 'VB', 'MD', 'VP', 'S',
                             'SINV', 'ADJP', 'NML', 'NP']),
              'SQ': ('l', ['VBZ', 'VBD', 'VBP', 'VB', 'MD', 'VP', 'SQ']),
              'VP': ('l', ['TO', 'VBD', 'VBN', 'MD', 'VBZ', 'VB', 'VBG', 'VBP',
                           'VP', 'ADJP', 'NN', 'NNS', 'NML', 'NP']),
              'WHADJP': ('l', ['CC', 'WRB', 'JJ', 'ADJP']),
              'WHADVP': ('r', ['CC', 'WRB']),
              'WHNP': ('l', ['WDT', 'WP',
                             'NML', 'NP', # RJ
                             'WP$', 'WHADJP', 'WHPP', 'WHNP']),
              'WHPP': ('r', ['IN', 'TO', 'FW']),

              'TOP': ('r', []),
              'ADVP|PRT': ('r', []),
              'META': ('r', []),
              'X': ('r', [])}

def coord_head(lbl, children):
    # unlike Collins, the coordinator is the head
    cs = children[1:]
    h = head_child('CC', cs)
    if h:
        return h
    h = head_child('CONJP', cs)
    if h:
        return h
    if lbl == 'UCP':
        return children[0]
    return None

def np_head(lbl, children):
    # unlike Collins, POS will not be the head
    inv = list(reversed(children))
    for c in inv:
        if get_label(c) in ['NN', 'NNP', 'NNPS', 'NNS', 'NX', 'JJR']:
            return c
    for c in children:
        if get_label(c) in ['NP', 'NML']:
            return c
    for c in inv:
        if get_label(c) in ['$', 'ADJP', 'PRN']:
            return c
    for c in inv:
        if get_label(c) == 'CD':
            return c
    for c in inv:
        if get_label(c) in ['JJ', 'JJS', 'RB', 'QP']:
            return c
    return children[-1]

def head_child(clbl, children):
    for c in children:
        if get_label(c) == clbl:
            return c
    return None
           
def find_head(lbl, children):
    ch = coord_head(lbl, children)
    if ch:
        return ch
    elif lbl in ['NP', 'NML', 'NX', 'NAC']:
        return np_head(lbl, children)
    else:
        dr, clbls = RULE_TABLE[lbl]
        if dr == 'l':
            children = list(reversed(children))
        for clbl in clbls:
            c = head_child(clbl, children)
            if c:
                return c
        return children[0]

def attach_head_labels(t):
    if isinstance(t, Tree):
        set_label(t, get_label(t) + ":[" + t.head[0] + "]")
        for c in t:
            attach_head_labels(c)

def set_heads(tree):
    if not isinstance(tree, Tree):
        return
    if not isinstance(tree[0], Tree):
        if len(tree.funtags) > 0:
            raise Exception("???")
        tree.head_child = tree
        tree.head = tree
        return
    for subtree in tree:
        set_heads(subtree)
    #tree.head_child = find_head(tree.label(), list(tree))
    tree.head_child = find_head(get_label(tree), list(tree))
    tree.head = tree.head_child.head

