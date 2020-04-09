# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 08:13:15 2020

@author: Felix
"""

# import libraries
import re
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#different classifier
from sklearn.ensemble import RandomForestClassifier

# Master data
db_place = 'sqlite:///.\data\P2_Wagener_disaster_response.db'
table_name = 't_clean_messages'
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(db_place, table_name):
    '''
    This function loads the data form the database
    
    It splits the function in feature columns and lable columns
    
    INPUT:
        master data:
            db_place : path where the db is stored
            table_name :  name of the table
        
    OUTPUT:
        X: data frame with features
        Y: data frame with lables
    '''
    engine = create_engine(db_place)
    df = pd.read_sql_table(table_name, engine)
    X = df['message']
    Y = df[df.columns[4:]]
    return X, Y



def tokenize(text):
    '''
    This function tokenizes the data
    It also replaces URLs by 'thiswasanurlbefore'
    
    INPUT:
        text = array of text
        
    OUTPUT:
        tokens =  text split up in tokens as list
    '''
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "thiswasanurlbefore")
    tokens = word_tokenize(text.lower().strip())
    return tokens

def delete_stop_words(text):
    '''
    This function delets stop words
    
    INPUT:
        text = list of tokens
        
    OUTPUT:
        text_no_stopwords =  list of tokens without stopwords
    '''
    text_no_stopwords = [word for word in text if word not in stopwords.words("english")]
    return text_no_stopwords

def delete_punctuation(text):
    '''
    This function delets the punctuation from tokens respectively tokens that 
    contain only punctuation
    
    INPUT:
        text = list of tokens
        
    OUTPUT:
        text_no_punctuation =  list of tokens without punctuation
    
    '''
    text_no_punctuation = re.sub(r'[^a-zA-z]', " ", text)
    return text_no_punctuation

def stemm(text):
    '''
    This function stemms the tokens
    
    INPUT:
        text = list of tokens
        
    OUTPUT:
        text_no_punctuation =  list of stemmed tokens
    '''
    text_stemmed = [PorterStemmer().stem(word) for word in text]
    return text_stemmed

def lematize(text):
    '''
    This function lematizes the tokens. It works on:
        a : ADJ
        r : ADV
        n : NOUN
        v : VERB
    
    INPUT:
        text = list of tokens
        
    OUTPUT:
        text_lematized =  list of lematized tokens       
    '''
    text_lematized = [WordNetLemmatizer().lemmatize(w, pos='a') for w in text]
    text_lematized = [WordNetLemmatizer().lemmatize(w, pos='r') for w in text_lematized]
    text_lematized = [WordNetLemmatizer().lemmatize(w, pos='n') for w in text_lematized]
    text_lematized = [WordNetLemmatizer().lemmatize(w, pos='v') for w in text_lematized]
    return text_lematized

def clean_text(text):
    '''
    This function combines some functions to fully clean up a given text
    
    INPUT:
        text - uncleaned string of data
        
    OUTPUT:
        clean_tokens_in_one_string - all tokens of a message in one string 
        devided by a space
    '''
    clean_tokens_in_one_string = []
    for message in text:
        message = delete_punctuation(message)
        message = tokenize(message)
        message = delete_stop_words(message)
        message = stemm(message)
        message = lematize(message)
        clean_tokens_in_one_string.append(' '.join(map(str, message)))

    return clean_tokens_in_one_string


def build_model():
    '''
    This function builds the model
    
    This function outputs the model that can be used to fit and predict data
    
    The transformer, estemator etc are stored in a pipeline
    
    GridSearchCV is included to search some parametersets and to finde best
    fitting parameters
    
    INPUT:
        --
        
    OUTPUT:
        model - model to fit and predict on text messages
        
    '''
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
                RandomForestClassifier()
                ))      
        ])
    
    # parameters to grid search
    parameters = { 'vectorizer__max_features' : [50],
            'clf__estimator__n_estimators' : [50] }

        
    # initiating GridSearchCV method
    model = GridSearchCV(pipeline, param_grid=parameters, cv = 5)

    return model
    
def main():
    
    # Import data
    print('Load data...')
    X, Y = load_data(db_place, table_name)
    print('...done')

    # Make it short and fast for testing
    X = X[0:10]
    Y = Y.head(10)

    # Clean data
    print('Cleaning Messages...')
    X = clean_text(X)
    print('...done')


    print('Split train and test data...')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
    print('...done')

    print('Build model...')
    model = build_model()
    print('...done')

    print('Fit model...')
    model.fit(X_train, y_train)
    print('...done')

    print('Evaluate model...')
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean().mean()
    print('The accuracy is:', accuracy)
    print('...done')


if __name__ == '__main__':
    main()