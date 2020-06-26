# PythonProject-June-2020

The goal of this project was to make a topic model of Saturday Night Live monologues to see what the most commonly discussed topics are.
The data I used was obtained from https://snltranscripts.jt.org/ which has transcripts of many monologues and sketches from SNL
The way this program works is that by using a loop, it opens all of the monologue transcripts and puts them into one large document.
Then the program throws away stopwords, words that do not contribute to the topic, punctuation and new line code.
The remaining words are put into a list with each word being an element of the list.
Then the program uses the bigram function from gensim to interpret two words that appear together often as one single token.
Once the data has been preprocessed, the program makes a corpus that gives each token a value and maps that value to the ammount of times the token appears in the text.
Then I ran three different topic modelling algorithms, LSI, HDP and LDA.
After getting my topic models, I used the Coherence Model function to see which algorithm produced the most coherent topic model for this data set and plotted the results.
For this dataset, the HDP model was the most coherent.

In order to run this program, you might have to change the path on lines 52 and 53 to open the data depending on where you save the txt files.
