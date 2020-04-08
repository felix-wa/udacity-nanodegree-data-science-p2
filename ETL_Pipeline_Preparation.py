# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 08:14:06 2020

@author: Felix
"""

# import libraries
import pandas as pd
from sqlalchemy import create_engine

# load messages dataset
messages = pd.read_csv('.\data\messages.csv')
messages.head()

# load categories dataset
categories = pd.read_csv('.\data\categories.csv')
categories.head()

# merge datasets
df = pd.merge(messages, categories, how = 'inner', left_on ='id', right_on = 'id')
df.head()

# create a dataframe of the 36 individual category columns
categories =  df['categories'].str.split(';', expand=True)
categories.head()

# select the first row of the categories dataframe
row = list(categories.iloc[0])
new_row = []
# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
for item in row:
    item = item.replace(r'-1', '').replace(r'-0', '')
    new_row.append(item)
row = new_row

# rename the columns of `categories`
categories.columns = row
categories.head()

for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str.strip().str[-1]
    
   # convert column from string to numeric
    pd.to_numeric(categories[column])
categories.head()

# drop the original categories column from `df`
df = df.drop(columns=['categories'])

df.head()

# concatenate the original dataframe with the new `categories` dataframe
#categories.reset_index(drop=True, inplace=True)
#df.reset_index(drop=True, inplace=True)

df = pd.concat([df, categories], axis=1, sort=False)
df

# check number of duplicates
#TRUE = duplicated ROWS
df.duplicated().value_counts()

# drop duplicates
df = df.drop_duplicates()

# check number of duplicates
df.duplicated().value_counts()

engine = create_engine('sqlite:///.\data\P2_Wagener_disaster_response.db')
df.to_sql('t_clean_messages', engine, index=False)