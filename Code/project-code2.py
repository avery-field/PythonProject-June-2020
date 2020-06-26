# imports
import matplotlib.pyplot as plt
import gensim
import numpy as np
import spacy

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
import pyLDAvis.gensim

import os, re, operator, warnings

# set spaCy to English
nlp = spacy.load("en")

# add my stopwords
my_stop_words = ['thing', 'slow', 'happy', 'change', 'know', 'think', 'want', 'happen',
                 'SNL' ,'play', 'cheer', 'say', 'come', 'get', 'right', 'time', 'Hill', 'Switon',
                 'coz', 'go', 'Amy', 'thank', 'oh', 'tell', 'yeah', 'walk', 'good', 'great',
                 'tina', 'laugh', 'feel', 'like', 'cheer', 'hey', 'to', 'okay', 'monologue',
                 'like', 'cut', 'start', 'let', 'look', 'applause', 'intro','Bobby', 'Moynihan',
                 'Alec', 'Baldwin', 'Kristen', 'Wiig', 'James', 'Franco', 'Seth', 'Rogan',
                 'Benedict', 'Cumberbatch', 'audience', 'Chris', 'Hemsworth', 'stage', 'Scarlett',
                 'Johansson', 'kind', 'Kyle', 'Adam', 'Driver', 'mean', 'Kenan', 'Thompson',
                 'Emma', 'Stone', 'Emily', 'Blunt', 'leave', 'host', 'Kate', 'McKinnon', 'Taraji', 
                 'P', 'announcer', 'little', 'John', 'Cena', 'Adams', 'fey', 'lady', 'hader', 'Jonah'
                 'gentleman', 'Melissa', 'McCarthy', 'Bill', 'Hader', 'Harvey', 'Feirstein', 
                 'Julia', 'Louis', 'Amy', 'Poehler', 'J.K.', 'Simmons', 'stick', 'Felicity', 
                 'Jones', 'Brie', 'Larson', 'Ronda', 'Rousey', 'Margot', 'Robbie', 'alright', 'Tilda',
                 'Dreyfus', 'gentleman', 'yes', 'Sarah', 'Silverman', 'Dwayne', 'Johnson', 'Bradley',
                 'people', 'Elizabeth', 'Banks', 'Callum', 'Lin', 'Manuel', 'Miranda', 'Leslie',
                 'Pete', 'pete', 'narrate', 'guy', 'leslie', 'Jim', 'P.', 'Henson', 'Aidy', 'Bryant',
                 'Stewart', 'try', 'sure', 'Ryan', 'Gosling', 'Davidson', 'Ariana', 'Grande', 
                 'cameron', 'diaz', 'see', 'lot', 'wanna', 'Jimmy', 'Fallon', 'Mike', 'Myers',
                 'camera', 'Live', 'give', 'stuff', 'need', 'fine', 'kid', 'way', 'Jonah', 'big',
                 'Larry', 'David', ' stay', 'Saturday', 'Night', 'chad', 'house', 'away', 'ha',
                 'Beck', 'Steve', 'Martin', 'Woody', 'Harrelson', 'Fierstein', 'Wooderson',
                 'um', 'talk', 'line', 'call', 'sorry', 'Carrey', 'Russell', 'Crowe', 'make',
                 'Goodman', 'Cecily', 'actually', 'everybody', 'molester', 'child', 'wow', 
                 'Tony', 'live', 'cool', 'care', 'main', 'strong', 'close', 'band', 'room',
                 'Tracy', 'Morgan', 'C.K.', 'Miley', 'Cyrus', 'McConaughey', 'Casey', 'Affleck',
                 'Venessa', 'Kevin', 'Hart', 'Dave', 'Chappelle', 'pa', 'stay', 'Tom', 'Hanks',
                 'Brad', 'Helvis', 'Lindsay', 'Parsin', 'Sasheer', 'Zamata', 'Taran', 'Killam', 'End']
for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True
    

# open txt files and put them in a doc
full_text = ""
for filename in os.listdir(os.path.join(os.getcwd(), "../Data/txt_files")):
    current_file = open("../Data/txt_files/" + filename, "r", errors = "ignore")
    full_text = full_text + current_file.read() + "*\n"
my_doc = nlp(full_text)
#print(my_doc)
texts, script, skl_texts = [], [], []
for w in my_doc:
    
# if it's not a stop word or punctuation mark, add it to script
# if the word or the lematized form of a word is a stop word then the word gets eliminated
    if w.text != '\n' and w.text != '\n\n' and not w.is_stop and w.lemma_ not in my_stop_words and w.lower_ not in my_stop_words and not w.is_punct and not w.like_num:
    #add lematized version of the word
        script.append(w.lemma_)
# if it's an asterisk, it means we're onto our next script
    if w.text == '*':
        skl_texts.append(' '.join(script))
        texts.append(script)
        script = []
#print(texts)

# make words that appear together often into a single token. ex. 'Merry', 'Christmas' becomes 'Merry_Christmas
bigram = gensim.models.Phrases(texts)
texts = [bigram[line] for line in texts]
#print(texts)
dictionary = Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]


# trying different topic modelling algorithms

lsimodel = LsiModel(corpus=corpus, num_topics=10, id2word=dictionary)
print("LSI Model:")
print(lsimodel.show_topics(num_topics=10))

hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)
print("HDP Model:")
print(hdpmodel.show_topics())

ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)
print("LDA Model:")
print(ldamodel.show_topics())

# using pyLDAvis to see which topic model is the most coherent for this data set

#pyLDAvis.enable_notebook()
#pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

lsitopics = [[word for word, prob in topic] for topicid, topic in lsimodel.show_topics(formatted=False)]

hdptopics = [[word for word, prob in topic] for topicid, topic in hdpmodel.show_topics(formatted=False)]

ldatopics = [[word for word, prob in topic] for topicid, topic in ldamodel.show_topics(formatted=False)]

# get coherence value

lsi_coherence = CoherenceModel(topics=lsitopics[:10], texts=texts, dictionary=dictionary, window_size=10).get_coherence()

hdp_coherence = CoherenceModel(topics=hdptopics[:10], texts=texts, dictionary=dictionary, window_size=10).get_coherence()

lda_coherence = CoherenceModel(topics=ldatopics, texts=texts, dictionary=dictionary, window_size=10).get_coherence()

# plot bar graph

def evaluate_bar_graph(coherences, indices):
    """
    Function to plot bar graph.
    
    coherences: list of coherence values
    indices: Indices to be used to mark bars. Length of this and coherences should be equal.
    """
    assert len(coherences) == len(indices)
    n = len(coherences)
    x = np.arange(n)
    plt.bar(x, coherences, width=0.2, tick_label=indices, align='center')
    plt.xlabel('Models')
    plt.ylabel('Coherence Value')
    
evaluate_bar_graph([lsi_coherence, hdp_coherence, lda_coherence],
                   ['LSI', 'HDP', 'LDA'])





