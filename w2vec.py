#! /usr/bin/env python

# import logging, gensim, bz2
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load id->word mapping (the dictionary), one of the results of step 2 above
# id2word = gensim.corpora.Dictionary.load_from_text('wiki_farsi_wordids.txt.bz2')
# load corpus iterator
# mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
# mm = gensim.corpora.MmCorpus(bz2.BZ2File('wiki_en_tfidf.mm.bz2')) # use this if you compressed the TFIDF output (recommended)

# print(mm)


# extract 400 LSI topics; use the default one-pass algorithm
# lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, num_topics=400)

# print the most contributing words (both positively and negatively) for each of the first ten topics
# lsi.print_topics(10)


# from gensim.models.word2vec import BrownCorpus, Word2Vec, LineSentence	

# sentences = LineSentence('wiki_farsi_wordids.txt.bz2')
# model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
# model.train()
# model.save('model_farsi')
# model = Word2Vec.load(fname)  # you can continue training with the loaded model!


from gensim.corpora import WikiCorpus
import logging, os. os.path
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
corpus = WikiCorpus('../itwiki-latest-pages-articles.xml.bz2',dictionary=False)
max_sentence = -1

def generate_lines():
    for index, text in enumerate(corpus.get_texts()):
        if index < max_sentence or max_sentence==-1:
            yield text
        else:
            break

from gensim.models.word2vec import BrownCorpus, Word2Vec
model = Word2Vec()
# Check if model is not exist
if ((os.path.exists('../model_farsi')) && (os.path.isfile('../model_farsi'))):
	model.Word2Vec.load('../model_farsi')
	result = model.most_similar(positive=[u'زن', u'پادشاه'], negative=[u'مرد'])
	print "result is:"
	print result

else:
	model.build_vocab(generate_lines())
	model.train(generate_lines(),chunksize=500)
	model.save(model_farsi)