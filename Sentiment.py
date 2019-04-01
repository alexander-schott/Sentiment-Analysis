from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import nltk
import csv
#nltk.download('subjectivity')




# read valence, arousal and text from dataset.
# divide into positive and negative sets, to ensure training and testing sets have equal ratios of pos and neg values
# ratio of training entries to testing entries. (want more to train than test)
train_test_ratio = 3/4
pos_sv = []
neg_sv = []
pos_sa = []
neg_sa = []
with open('emobank.csv') as eb:
    reader = csv.DictReader(eb)
    try:
        for row in reader:

                v = float(row["V"])
                a = float(row["A"])
                txt = nltk.word_tokenize(row['text'])
                # change from 6 pt positive scale to 10 point scale centered at 0
                v = v * 10 / 6 - 5
                a = a * 10 / 6 - 5
                #print(v, a, txt)

                #valence
                if v > 0:
                    # pos_sv.append((txt, v))
                    pos_sv.append((txt, 'pos'))
                elif v < 0:
                    #neg_sv.append((txt, v))
                    neg_sv.append((txt, 'neg'))
                else:
                    # v == 0
                    # if value is neutral want to more or less randomly pick destination
                    if len(txt) % 2 == 0:
                        #pos_sv.append((txt, v))
                        pos_sv.append((txt, 'pos'))
                    else:
                        #neg_sv.append((txt, v))
                        neg_sv.append((txt, 'neg'))
                # Arousal
                if a > 0:
                    pos_sa.append((txt, a))
                elif a < 0:
                    neg_sa.append((txt, a))
                else:
                    # a == 0
                    if len(txt) % 2 == 0:
                        pos_sa.append((txt, a))
                    else:
                        neg_sa.append((txt, a))
    except ValueError:
        print("Invalid character. Continuing")



val_train_docs = []
val_test_docs = []
ar_train_docs = []
ar_test_docs = []
#divide pos_sv
for i in range(0, (int) (len(pos_sv) * train_test_ratio)):
    val_train_docs.append(pos_sv[i])
for i in range((int)(len(pos_sv) * train_test_ratio), len(pos_sv)):
    val_test_docs.append(pos_sv[i])
    
#divide neg sv
for i in range(0, (int)(len(neg_sv) * train_test_ratio)):
    val_train_docs.append(neg_sv[i])
for i in range((int)(len(neg_sv) * train_test_ratio), len(neg_sv)):
    val_test_docs.append(neg_sv[i])

#divide pos sa
for i in range(0, (int)(len(pos_sa) * train_test_ratio)):
    ar_train_docs.append(pos_sa[i])
for i in range((int)(len(pos_sa) * train_test_ratio), len(pos_sa)):
    ar_test_docs.append(pos_sa[i])

# divide neg sa
for i in range(0, (int)(len(neg_sa) * train_test_ratio)):
    ar_train_docs.append(neg_sa[i])
for i in range((int)(len(neg_sa) * train_test_ratio), len(neg_sa)):
    ar_test_docs.append(neg_sa[i])


valence_analyzer = SentimentAnalyzer()
arousal_analyzer = SentimentAnalyzer()

dictionary = {}
for txt, val in val_train_docs:
    #print(txt)
    for word in txt:
        dictionary[word.lower()] = 1
valence_training_set = [({word: (word in x[0]) for word in dictionary}, x[1]) for x in val_train_docs]

#build set with features
#valence_training_set = valence_analyzer.apply_features(val_train_docs)
valence_test_set = valence_analyzer.apply_features(val_test_docs)
arousal_training_set = arousal_analyzer.apply_features(val_train_docs)
arousal_test_set = arousal_analyzer.apply_features(val_test_docs)

#train and test
trainer = NaiveBayesClassifier.train

#train
#valence_classifier = valence_analyzer.train(trainer, valence_training_set)
valence_classifier = nltk.NaiveBayesClassifier.train(valence_training_set)
#test

#for key,value in sorted(valence_analyzer.evaluate(valence_test_set).items()):
#     print('{0}: {1}'.format(key, value))


arousal_classifier = arousal_analyzer.train(trainer, arousal_training_set)
#for key,value in sorted(arousal_analyzer.evaluate(arousal_test_set).items()):
#     print('{0}: {1}'.format(key, value))

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

test_docs = valence_analyzer.apply_features(val_test_docs)
for sent in test_docs:
    print(sent)
    print(sent
          , "    valence: ", valence_classifier.classify(sent)
          , "    arousal: ", arousal_classifier.classify(sent))
"""

for sentence, val in val_test_docs:
    test_data_features = []
    for word in (sentence):
        if word in dictionary:
         test_data_features.append(word)
    sentence = (test_data_features)
    print(sentence
          , "    valence: ", valence_classifier.classify(test_data_features)
          , "    arousal: ", arousal_classifier.classify(test_data_features))
"""

"""
for sentence in sentences:
    # sentence = nltk.word_tokenize(sentence)
    print(sentence
          , "    valence: ", vc.classify(sentence)
          , "    arousal: ", ac.classify(sentence))
"""
