#! /usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.corpora import WikiCorpus
import logging, os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.models.word2vec import Word2Vec
model = Word2Vec()
# Check if model is not exist
if ((os.path.exists('model_farsi')) and (os.path.isfile('model_farsi'))):
	model.load('model_farsi')
	result = model.most_similar(positive=[u'زن', u'مرد'], negative=[u''], topn=1)
	
	print "result is:"
	print result

else:
	

	print "File does not exist"