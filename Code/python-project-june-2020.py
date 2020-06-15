#imports
import matplotlib.pyplot as plt
import gensim
import numpy as np
import spacy

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
import pyLDAvis.gensim

import os, re, operator, warnings

from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS


nlp = English()

#data
file = open("../Data/txt_files/drake.txt", "r", errors = "ignore")
full_text = file.read()
my_doc = nlp(full_text)


#remove stopwords
token_list = []
for token in my_doc:
    token_list.append(token.text)
    

STOP_WORDS.add("audience")
STOP_WORDS.add("monologue")
STOP_WORDS.add("guest")
STOP_WORDS.add("ladies")
STOP_WORDS.add("gentlemen")
STOP_WORDS.add("host")
STOP_WORDS.add("if")
STOP_WORDS.add("and")
STOP_WORDS.add("the")
STOP_WORDS.add("Announcer")

#print(list(nlp.vocab.strings))
for word in STOP_WORDS:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True
    
filtered_sentence =[] 

for word in token_list:
    lexeme = nlp.vocab[word]
    if lexeme.is_stop == False:
        filtered_sentence.append(word) 
#print(token_list)
#print(filtered_sentence)

texts, article, skl_texts = [], [], []
for w in my_doc:
    # if it's not a stop word or punctuation mark, add it to our article!
    if w.text != '\n' and w.text != '\n\n' and not w.is_stop and not w.is_punct and not w.like_num:
        # we add the lematized version of the word
        article.append(w.lemma_)
    # if it's a new line, it means we're onto our next document
    
skl_texts.append(' '.join(article))
texts.append(article)


bigram = gensim.models.Phrases(texts)
texts = [bigram[line] for line in texts]

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)

ldamodel = LdaModel(corpus=corpus, num_topics=3, id2word=dictionary)
print(ldamodel.show_topics())


