# import libraries
import sys
import re
import pickle
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Master data
db_place = 'sqlite:///.\P2_disaster_response.db'#'sqlite:///.\data\P2_disaster_response.db'
table_name = 't_clean_messages'
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(db_place):
    '''
    This function loads the data form the database
    
    It splits the function in feature columns and lable columns
    
    INPUT:
        master data:
            db_place : path where the db is stored
            table_name :  name of the table
        
    OUTPUT:
        X : data frame with features
        Y : data frame with lables
        category_names : list of category names
    '''
    db_name = 'sqlite:///.\\' + db_place
    engine = create_engine(db_name)
    df = pd.read_sql_table('t_clean_messages', engine)
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = list(Y.columns)
    return X, Y, category_names


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
    text_no_stopwords =[]
    for word in text:
        if word not in stopwords.words("english"):
            text_no_stopwords.append(word)
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
        text : clean text in tokens
    '''
    text = delete_punctuation(text)
    text = tokenize(text)
    text = delete_stop_words(text)
    #text = stemm(text)
    text = lematize(text)
    return text


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
        model : model to fit and predict on text messages
        
    '''
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=clean_text)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
                AdaBoostClassifier()#RandomForestClassifier()
                ))      
        ])
    
    # parameters to grid search
    parameters = { 'vectorizer__max_features' : [300],#, 72, 144, 288, 576, 1152],
            'clf__estimator__n_estimators' : [50]}#, 50, 100, 125, 150] }
   
    # initiating GridSearchCV method
    model = GridSearchCV(pipeline, param_grid=parameters, cv = 5)
    
    return model


def evaluate_model(model, X_test, y_test, category_names):
    '''
    This function evaluates the ML model and prints evaluation metrics
    
    INPUT:
        model : trained model, that should be investigated
        X_text : feature values that should be used for evaluation
        y_test : real lable values for evaluation of prediction
        category_names : names of all categories
    OUTPUT:
        no return value
        different metrics and the overall metric are printed
    '''
    # predict test data
    y_pred = model.predict(X_test)
    for category in range(len(category_names)):
        print("Category:", category_names[category],"\n", classification_report(y_test.iloc[:, category].values, y_pred[:, category]))
        print('Accuracy of', (category_names[category]), 'is', accuracy_score(y_test.iloc[:, category].values, y_pred[:,category]))
        print('')
    accuracy = (y_pred == y_test).mean().mean()
    print('')
    print('The overall accuracy is:', accuracy)

def save_model(model, model_filepath):
    '''
    This function saves the ML model
    
    INPUT:
        model : trained model
        model_filepath : path where the trainded model should be saved on disc
    OUTPUT:
        no return value
        trained model is saved to disc
    '''
    pickle.dump(model, open(model_filepath, 'wb'))




def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        print('...done')
        
        #For tests comment in have quick runs with less data
        X = X.head(1000)
        Y = Y.head(1000)
                
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
        
        print('Building model...')
        model = build_model()
        print('...done')
        
        print('Training model...')
        model.fit(X_train, Y_train)      
        print('...done')
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        print('...done')

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('...done. Trained model saved!')
        
        print(X)
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()