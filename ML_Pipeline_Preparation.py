# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 08:13:15 2020

@author: Felix
"""

# import libraries
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



#import nltk
##nltk.download(['punkt', 'wordnet', 'stopwords'])


# load data from database
db_place = 'sqlite:///.\data\P2_Wagener_disaster_response.db'
table_name = 't_clean_messages'
def load_data(db_place, table_name):
    engine = create_engine(db_place)
    df = pd.read_sql_table(table_name, engine)
    X = df.message.values
    Y = df[df.columns[4:]]
    return X, Y



def tokenize(text):
    #INPUT text = array of text
    #OUTPUT tokens =  text split up in tokens as list
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "thiswasanurlbefore")
    tokens = word_tokenize(text.lower().strip())
    return tokens

def delete_stop_words(text):
    #INPUT text = list of tokens
    #OUTPUT text_no_stopwords =  list of tokens without stopwords
    text_no_stopwords = [word for word in text if word not in stopwords.words("english")]
    return text_no_stopwords

def delete_punctuation(text):
    #INPUT text = list of tokens
    #OUTPUT text_no_punctuation =  list of tokens without punctuation
    text_no_punctuation = re.sub(r'[^a-zA-z]', " ", text)
    return text_no_punctuation

def stemm(text):
    #INPUT text = list of tokens
    #OUTPUT text_no_punctuation =  list of stemmed tokens
    text_stemmed = [PorterStemmer().stem(word) for word in text]
    return text_stemmed

def lematize(text):
    #INPUT text = list of tokens
    #OUTPUT text_lematized =  list of lematized tokens
    text_lematized = [WordNetLemmatizer().lemmatize(w, pos='a') for w in text]
    text_lematized = [WordNetLemmatizer().lemmatize(w, pos='r') for w in text_lematized]
    text_lematized = [WordNetLemmatizer().lemmatize(w, pos='n') for w in text_lematized]
    text_lematized = [WordNetLemmatizer().lemmatize(w, pos='v') for w in text_lematized]
    return text_lematized

def clean_text(text):
    clean_tokens = []
    for message in text:
        #print('Delete punctuation...')
        message = delete_punctuation(message)
        #print('...done')
        #print('tokenziz...')
        message = tokenize(message)
        #print('...done')
        #print('Delete stop words...')
        message = delete_stop_words(message)
        #print('...done')
        #print('Stemm...')
        message = stemm(message)
        #print('...done')
        #print('Lematize...')
        message = lematize(message)
        #print('...done')
        clean_tokens.append(' '.join(map(str, message)))
        #clean_tokens.append(message)
    return clean_tokens


# importing parameter
db_place = 'sqlite:///.\data\P2_Wagener_disaster_response.db'
table_name = 't_clean_messages'

#import data
print('Load data...')
X, Y = load_data(db_place, table_name)
print('...done')
#make it short and fast
#X = X[0:5000]
#Y = Y.head(5000)
print('Cleaning Messages...')
X = clean_text(X)
print('...done')

vectorizer = CountVectorizer(max_features = 2000)
tfidf = TfidfTransformer()
forest = RandomForestClassifier(n_estimators=100)
clf = MultiOutputClassifier(forest)

print('Vectorize data...')
X_counts = vectorizer.fit_transform(X)
print('...done')

print('Split train and test data...')
X_train_count, X_test_count, y_train, y_test = train_test_split(X_counts, Y, test_size = 0.2, random_state = 42)
print('...done')

print('Transform data...')
X_train_tfidf = tfidf.fit_transform(X_train_count)
X_test_tfidf = tfidf.fit_transform(X_test_count)
print('...done')

#print(X_train_tfidf.shape)
#print(y_train.shape)

print('Training model...')
clf.fit(X_train_tfidf, y_train)
print('...done')

print(X_test_tfidf.shape)
print(y_test.shape)
print('Predict values...')
y_pred = clf.predict(X_test_tfidf)
print('...done')

score = clf.score(X_test_tfidf, y_test)
print(score)


