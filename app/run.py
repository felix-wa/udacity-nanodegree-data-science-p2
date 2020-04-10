import json
import pandas as pd
import plotly
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def tokenize(text):
    '''
    This function tokenizes the data
    It also replaces URLs by 'thiswasanurlbefore'
    
    INPUT:
        text : array of text
        
    OUTPUT:
        tokens :  text split up in tokens as list
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
        text : list of tokens
        
    OUTPUT:
        text_no_stopwords : list of tokens without stopwords
    '''
    text_no_stopwords = [word for word in text if word not in stopwords.words("english")]
    return text_no_stopwords

def delete_punctuation(text):
    '''
    This function delets the punctuation from tokens respectively tokens that 
    contain only punctuation
    
    INPUT:
        text : list of tokens
        
    OUTPUT:
        text_no_punctuation :  list of tokens without punctuation
    
    '''
    text_no_punctuation = re.sub(r'[^a-zA-z]', " ", text)
    return text_no_punctuation

def stemm(text):
    '''
    This function stemms the tokens
    
    INPUT:
        text : list of tokens
        
    OUTPUT:
        text_no_punctuation :  list of stemmed tokens
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
        text : list of tokens
        
    OUTPUT:
        text_lematized : list of lematized tokens       
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
        text : uncleaned string of data
        
    OUTPUT:
        clean_tokens_in_one_string : all tokens of a message in one string 
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
# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('t_clean_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_names = df.iloc[:,4:].columns
    category_values = []
    for column in category_names:
        df[column] = pd.to_numeric(df[column])
        category_values.append(df[column].sum())
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                    )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_values,
                )
            ],

            'layout': {
                'title': 'Distribution of Message In learing and test dataset',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()