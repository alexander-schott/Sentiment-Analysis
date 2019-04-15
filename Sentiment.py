from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
#import csv
import pandas as pd
#nltk.download('subjectivity')
#nltk.download('stopwords')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


"happy, sad, angry, calm"

"""
Function to reduce the number of words in the programs dictionary. 
Converts sentence to a list of lemmatized lowercase words.
"""


sentences = ["VADER is smart, handsome, and funny.", # positive sentence example
    "VADER is smart, handsome, and funny!", # punctuation emphasis handled correctly (sentiment intensity adjusted)
    "VADER is very smart, handsome, and funny.",  # booster words handled correctly (sentiment intensity adjusted)
    "VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
    "VADER is VERY SMART, handsome, and FUNNY!!!",# combination of signals - VADER appropriately adjusts intensity
    "VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!",# booster words & punctuation make this close to ceiling for score
    "The book was good.",         # positive sentence
    "The book was kind of good.", # qualified positive sentence is handled correctly (intensity adjusted)
    "The plot was good, but the characters are uncompelling and the dialog is not great.", # mixed negation sentence
    "A really bad, horrible book.",       # negative sentence with booster words
    "At least it isn't a horrible book.", # negated negative sentence with contraction
    ":) and :D",     # emoticons handled
    "",              # an empty string is correctly handled
    "Today sux",     #  negative slang handled
    "Today sux!",    #  negative slang with punctuation emphasis handled
    "Today SUX!",    #  negative slang with capitalization emphasis
    "Today kinda sux! But I'll get by, lol" # mixed sentiment example with slang and constrastive conjunction "but"
]
tricky_sentences = [
    "Most automated sentiment analysis tools are shit.",
    "VADER sentiment analysis is the shit.",
    "Sentiment analysis has never been good.",
    "Sentiment analysis with VADER has never been this good.",
    "Warren Beatty has never been so entertaining.",
    "I won't say that the movie is astounding and I wouldn't claim that \
    the movie is too banal either.",
    "I like to hate Michael Bay films, but I couldn't fault this one",
    "It's one thing to watch an Uwe Boll film, but another thing entirely \
    to pay for it",
    "The movie was too good",
    "This movie was actually neither that funny, nor super witty.",
    "This movie doesn't care about cleverness, wit or any other kind of \
    intelligent humor.",
    "Those who find ugly meanings in beautiful things are corrupt without \
    being charming.",
    "There are slow and repetitive parts, BUT it has just enough spice to \
    keep it interesting.",
    "The script is not fantastic, but the acting is decent and the cinematography \
    is EXCELLENT!",
    "Roger Dodger is one of the most compelling variations on this theme.",
    "Roger Dodger is one of the least compelling variations on this theme.",
    "Roger Dodger is at least compelling as a variation on the theme.",
    "they fall in love with the product",
    "but then it breaks",
    "usually around the time the 90 day warranty expires",
    "the twin towers collapsed today",
    "However, Mr. Carter solemnly argues, his client carried out the kidnapping \
    under orders and in the ''least offensive way possible.''"
]
sentences.append(tricky_sentences)
vader = SentimentIntensityAnalyzer()
for sentence in sentences:
    valence = vader.polarity_scores(sentence)['compound']
    print(sentence)
    print("vader: ", valence)

def sentenceToFeatures(sentence):
    lem = WordNetLemmatizer()
    txt = []
    for word in nltk.word_tokenize(sentence):
        word = word.lower()
        word = lem.lemmatize(word, "v")
        if word not in stopwords.words("english"):
            txt.append(word)
            # Would be time efficent to add words to dictionary here,
            # but to keep this function more general I will not.
            #dictionary.add(word)
    return txt

valence_docs = []
arousal_docs = []
sentiment_docs = []
test_docs = []
dictionary = set()
pv = 0
nv = 0
pa = 0
na = 0

def categorizeSentiment(v, a):
    sent = ""
    if v <= 0 and a <= 0:
        sent = "sad"
    if v <= 0 and a > 0:
        sent = "angry"
    if v > 0 and a <= 0:
        sent = "calm"
    if v > 0 and a > 0:
        sent = "happy"
    return sent

def addData(text, v, a):
    try:
        sent = categorizeSentiment(v, a)
        txt = sentenceToFeatures(text)
        for word in txt:
            dictionary.add(word)
        #test_docs.append(text)
        valence_docs.append((txt, v))
        arousal_docs.append((txt, a))
        sentiment_docs.append((txt, sent))
    except:
        print("Failed to add text: ", text)

print("Reading data...")
eb = pd.read_csv('emobank.csv', index_col=0)
nrc = pd.read_csv('NRC-VAD-Lexicon.txt', delim_whitespace=True, error_bad_lines=False)

#read nrc
for index, row in nrc.iterrows():
    word = row[0]
    v = float(row[1])
    a = float(row[2])
    v = v * 10 - 5
    a = a * 10 - 5
    if v >= 0:
        pv += 1
    else:
        nv += 1
    if a >= 0:
        pa += 1
    else:
        na += 1
    #addData(word, v, a)

#read emobank
test_count = 100
for index, row in eb.iterrows():
    v = float(row["V"])
    a = float(row["A"])
    text = row["text"]
    # change from 6 pt positive scale to 10 point scale centered at 0
    v = v * 10 / 6 - 5
    a = a * 10 / 6 - 5
    if v >= 0:
        pv += 1
    else:
        nv += 1
    if a >= 0:
        pa += 1
    else:
        na += 1

    test_count += 1
    if test_count % 100 == 0:
        test_docs.append(text)
    addData(text, v, a)


print("Number of entries with positive or neutral Valence: ", pv)
print("Number of entries with negative Valence: ", nv)
print("Number of entries with positive or neutral Arousal: ", pa)
print("Number of entries with negative Arousal: ", na)
print("Number of words in dictionary: ", len(dictionary))
print("Number of test docs: ", len(test_docs))
print("Building training sets...")
valence_training_set = [({word: (word in (x[0])) for word in dictionary}, x[1]) for x in valence_docs]
arousal_training_set = [({word: (word in (x[0])) for word in dictionary}, x[1]) for x in arousal_docs]
sentiment_training_set = [({word: (word in (x[0])) for word in dictionary}, x[1]) for x in sentiment_docs]

print("Training Valence Classifier...")
valence_classifier = NaiveBayesClassifier.train(valence_training_set)

print("Training Arousal Classifier...")
arousal_classifier = NaiveBayesClassifier.train(arousal_training_set)

print("Training Sentiment Classifier...")
sentiment_classifier = NaiveBayesClassifier.train(sentiment_training_set)

print("Testing...")


for sentence in sentences:
    test_data_features = {word: (word in sentenceToFeatures(sentence)) for word in dictionary}
    print("Sentence: ", sentence)
    print("Valence: ", valence_classifier.classify(test_data_features))
    print("Arousal: ", arousal_classifier.classify(test_data_features))
    print("Sentiment: ", sentiment_classifier.classify(test_data_features))



for sentence in test_docs:
    test_data_features = {word: (word in sentenceToFeatures(sentence)) for word in dictionary}
    print("Sentence: ", sentence)
    print("Valence: ", valence_classifier.classify(test_data_features))
    print("Arousal: ", arousal_classifier.classify(test_data_features))
    print("Sentiment: ", sentiment_classifier.classify(test_data_features))
