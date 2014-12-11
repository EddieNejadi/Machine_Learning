#! /usr/bin/env python
# -*- coding: utf-8 -*-



from gensim.corpora import WikiCorpus
from gensim.models.word2vec import Word2Vec
import logging, os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
corpus = WikiCorpus('../fawiki-latest-pages-articles.xml.bz2',dictionary=False)

max_sentence = -1

def generate_lines():
    for index, text in enumerate(corpus.get_texts()):
        if index < max_sentence or max_sentence==-1:
            yield text
        else:
            break

# Check if model is not exist
model = Word2Vec() 		
if ((os.path.exists('../model_farsi')) and (os.path.isfile('../model_farsi'))):
	model = Word2Vec.load('../model_farsi')
	result_1 = model.most_similar('روز')
	result_2 = model.most_similar(positive=['زن', 'پادشاه'], negative=['مرد'], topn=10)
	
	print "result is:"
	for (re,v) in result_1:
		print re + ' '+ str(v)
	print "======================="
	for (re,v) in result_2:
		print re + ' '+ str(v)


else:
	model.build_vocab(corpus.get_texts()) 
	model.train(generate_lines(),chunksize=500)
	model.save('../model_farsi')
