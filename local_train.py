#! /usr/bin/env python
# -*- coding: utf-8 -*-

max_sentence = -1
def generate_lines():
    for index, text in enumerate(corpus.get_texts()):
        if index < max_sentence or max_sentence==-1:
            yield text
        else:
            break

from gensim.corpora import WikiCorpus
import logging, os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.models.word2vec import Word2Vec
model = Word2Vec()
# Check if model is not exist
if ((os.path.exists('../model_farsi')) and (os.path.isfile('../model_farsi'))):
	model.load('../model_farsi')
	result = model.most_similar(u'زن')
	
	print "result is:"
	print result

else:
	corpus = WikiCorpus('../fawiki-latest-pages-articles.xml',dictionary=False)
	model = Word2Vec() 
	model.build_vocab(generate_lines()) #This strangely builds a vocab of "only" 747904 words which is << than those reported in the literature 10M words
	model.train(generate_lines(),chunksize=500)
	model.save('../model_farsi')
