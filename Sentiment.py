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


"""
Function to reduce the number of words in the programs dictionary. 
Converts sentence to a list of lemmatized lowercase words.
"""
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

# read valence, arousal and text from dataset.
# divide into positive and negative sets, to ensure training and testing sets have equal ratios of pos and neg values
# ratio of training entries to testing entries. (want more to train than test)
train_test_ratio = 3/4
valence_docs = []
arousal_docs = []
test_docs = []
dictionary = set()
pv = 0
nv = 0
pa = 0
na = 0


print("Reading data...")
eb = pd.read_csv('emobank.csv', index_col=0)
nrc = pd.read_csv('NRC-VAD-Lexicon.txt', index_col=False, skiprows=[0], delim_whitespace=True)
for index, row in nrc.iterrows():
    """
    word = row["Word"]
    v = float(row["Valence"])
    a = float(row["Arousal"])
    """
    print(row)
    print(index)
    word = index[0]
    v = float(index[1])
    a = float(index[2])
    print(word, v, a)
for index, row in eb.iterrows():
    v = float(row["V"])
    a = float(row["A"])
    # Text strings are converted to a list of lowercase words.
    txt = sentenceToFeatures(row['text'])
    for word in txt:
        dictionary.add(word)
    test_docs.append(row['text'])
    # change from 6 pt positive scale to 10 point scale centered at 0
    v = v * 10 / 6 - 5
    a = a * 10 / 6 - 5
    #print(v, a, txt)
    valence_docs.append((txt, v))
    arousal_docs.append((txt, a))
    if v >= 0:
        pv += 1
    else:
        nv += 1
    if a >= 0:
        pa += 1
    else:
        na += 1
    #except ValueError:
     #   print("Invalid character. Continuing")
print("Number of entries with positive or neutral Valence: ", pv)
print("Number of entries with negative Valence: ", nv)
print("Number of entries with positive or neutral Arousal: ", pa)
print("Number of entries with negative Arousal: ", na)
print("Number of words in dictionary: ", len(dictionary))
print("Number of test docs: ", len(test_docs))
print("Building training sets...")
valence_training_set = [({word: (word in (x[0])) for word in dictionary}, x[1]) for x in valence_docs]
arousal_training_set = [({word: (word in (x[0])) for word in dictionary}, x[1]) for x in arousal_docs]

print("Training Valence Classifier...")
valence_classifier = NaiveBayesClassifier.train(valence_training_set)

print("Training Arousal Classifier...")
arousal_classifier = NaiveBayesClassifier.train(arousal_training_set)

print("Testing...")
for sentence in test_docs:
    test_data_features = {word: (word in sentenceToFeatures(sentence)) for word in dictionary}
    print("Sentence: ", sentence)
    print("Valence: ", valence_classifier.classify(test_data_features))
    print("Arousal: ", arousal_classifier.classify(test_data_features))


