# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 08:14:06 2020

@author: Felix
"""

# import libraries
import pandas as pd
from sqlalchemy import create_engine


#master data
messages_filepath = '.\data\disaster_messages.csv'
categories_filepath = '.\data\disaster_categories.csv'
database_filename = 'sqlite:///.\data\P2_disaster_response.db'

def load_data(messages_filepath, categories_filepath):
    '''
    This function loads the relevant data from the CSV files, megers them and
    returns the megered dataframe
    
    INPUT:
        messages_filepath - path of messages CSV file
        categories_filepath - path of categories CSV file
        
    OUTPUT:
        df_merged - merged dataframe containing message and categorical data
    
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    messages.head()
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    categories.head()

    # merge datasets
    df_merged = pd.merge(messages, categories, how = 'inner', left_on ='id', right_on = 'id')
    return df_merged


def clean_data(df):
    '''
    This function cleans the data in several steps:
        - splitting the categorical column in 36 different columns that each
        contain the information of one lable
        - clear the column names
        - change entries to numeric values (0 or 1)
        
    INPUT:
        df - dataframe with messages and categories, where all categories are
        in one column
        
    OUTPUT:
        df - dataframe where categories are split in different columns and have
        numeric values
    '''
    # split category colums in 36 columns
    categories =  df['categories'].str.split(';', expand=True)

    # extract lable names from first row
    row = list(categories.iloc[0])
    new_row = []
    for item in row:
        item = item.replace(r'-1', '').replace(r'-0', '')
        new_row.append(item)
        row = new_row

    # rename the category columns by names we just extracted from the first row
    categories.columns = row

    #change entries of colums to numeric value (eg 'health_1' --> 1)
    for column in categories:
        categories[column] = categories[column].str.strip().str[-1]
        pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])

    # concatthe initial df with the new categories
    df = pd.concat([df, categories], axis=1, sort=False)

    # drop duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    '''
    This function saves the clean data in a SQL database
    
    INPUT:
        df - dataframe that should be saved
        database_filename - filepath where the DB will be located
        
    OUTPUT:
        no return value
        saved database 
    '''
    engine = create_engine(database_filename)
    df.to_sql('t_clean_messages', engine, index=False)
    
    
df = load_data(messages_filepath, categories_filepath)
df = clean_data(df)
save_data(df, database_filename)