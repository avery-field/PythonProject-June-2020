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

nlp = English()

#data
file = open("../Data/txt_files/drake.txt", "r", errors = "ignore")
full_text = file.read()
print(full_text)

my_doc = nlp(full_text)

#remove stopwords
token_list = []
for token in my_doc:
    token_list.append(token.text)
    
from spacy.lang.en.stop_words import STOP_WORDS

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

for word in STOP_WORDS:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True
    
filtered_sentence =[] 

for word in token_list:
    lexeme = nlp.vocab[word]
    if lexeme.is_stop == False:
        filtered_sentence.append(word) 
print(token_list)
print(filtered_sentence) 