import sys
import pandas as pd
from sqlalchemy import create_engine

##master data
#messages_filepath = '.\data\disaster_messages.csv'
#categories_filepath = '.\data\disaster_categories.csv'
#database_filename = 'sqlite:///.\data\P2_disaster_response.db'


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
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])

    # concatthe initial df with the new categories
    df = pd.concat([df, categories], axis=1, sort=False)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filepath):
    '''
    This function saves the clean data in a SQL database
    
    INPUT:
        df - dataframe that should be saved
        database_filepath - filepath where the DB will be located
        
    OUTPUT:
        no return value
        saved database 
    '''
    db_name = 'sqlite:///.\\' + database_filepath
    engine = create_engine(db_name)
    df.to_sql('t_clean_messages', engine, if_exists = 'replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print('...done')
        
        print('Cleaning data...')
        df = clean_data(df)
        print('...done')
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        print('...done')
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()