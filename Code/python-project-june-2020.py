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

STOP_WORDS.add("audience")
STOP_WORDS.add("cheers")
STOP_WORDS.add("like")
STOP_WORDS.add("applause")
STOP_WORDS.add("monologue")
STOP_WORDS.add("guest")
STOP_WORDS.add("ladies")
STOP_WORDS.add("gentlemen")
STOP_WORDS.add("host")
STOP_WORDS.add("Tina")
STOP_WORDS.add("Fey")
STOP_WORDS.add("Amy")
STOP_WORDS.add("Poehler")
STOP_WORDS.add("Announcer")
STOP_WORDS.add("Emma")
STOP_WORDS.add("Stone")
STOP_WORDS.add("Oh")
STOP_WORDS.add("Yeah")
STOP_WORDS.add("Hemsworth")
STOP_WORDS.add("ha")
STOP_WORDS.add("know")
STOP_WORDS.add("get")
STOP_WORDS.add("walk")
STOP_WORDS.add("Cut")
STOP_WORDS.add("Thanks")
STOP_WORDS.add("stage")
STOP_WORDS.add("right")
STOP_WORDS.add("thinks")
STOP_WORDS.add("play")
STOP_WORDS.add("Thanks")
STOP_WORDS.add("great")
STOP_WORDS.add("go")
STOP_WORDS.add("to")
STOP_WORDS.add("come")
STOP_WORDS.add("look")
STOP_WORDS.add("time")
STOP_WORDS.add("join")
STOP_WORDS.add("baby")
STOP_WORDS.add("change")
STOP_WORDS.add("Kenan")
STOP_WORDS.add("Taran")
STOP_WORDS.add("Bobby")
STOP_WORDS.add("glory")
STOP_WORDS.add("thing")
STOP_WORDS.add("play")


nlp = English()
texts, article, skl_texts = [], [], []
#data
for filename in os.listdir(os.path.join(os.getcwd(), "../Data/txt_files")):
    current_file = open("../Data/txt_files/" + filename, "r", errors = "ignore")
    full_text = current_file.read()
    my_doc = nlp(full_text)
    #print(my_doc)

#data
#file = open("../Data/txt_files/drake.txt", "r", errors = "ignore")
#full_text = file.read()
#my_doc = nlp(full_text)
#print(my_doc)

#remove stopwords
    token_list = []
    for token in my_doc:
        token_list.append(token.text)

    
   

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

    
for w in my_doc:
    # if it's not a stop word or punctuation mark, add it to our article!
    if w.text != '\n' and w.text != '\n\n' and not w.is_stop and not w.is_punct and not w.like_num:
        # we add the lematized version of the word
        article.append(w.lemma_)
    # if it's a new line, it means we're onto our next document
    
skl_texts.append(' '.join(article))
texts.append(article)
#print(article)

bigram = gensim.models.Phrases(texts)
texts = [bigram[line] for line in texts]

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
#print(corpus)
#print(texts)


ldamodel = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary)
print(ldamodel.show_topics())


