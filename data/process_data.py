import sys

#import libraries
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ 
    Load messages and categories csv and create a dataframe.
    
    Parameters: 
    messages_filepath (str): Path to messages.csv
    categories_filepath (str): Path to categories.csv
  
    Returns: 
    DataFrame: DataFrame with columns from both messages and categories
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id',how='outer')
    return df


def clean_data(df):
    """
    Function to clean disaster response data 

    Parameters: 
        df (DataFrame): DataFrame loaded in using loaddata() function
      
    Returns: 
        DataFrame: Cleaned data ready to be saved to database
    """
    categories = df['categories']
    categories = categories.str.split(';',expand=True)
    category_colnames = categories[:1].apply(lambda x: x.str.slice(0,-2,1)).values
    categories.columns = category_colnames[0]
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].apply(pd.to_numeric)
    df.drop(columns=['categories'],inplace=True)
    df = pd.concat([df,categories], axis=1, join="outer")
    df.drop_duplicates(inplace=True)
    df.loc[df.related == 2, 'related'] = 0
    return df

def save_data(df, database_filename):
    """
    Function to create a SQLite database and save cleaned data into that database 

    Parameters: 
        df (DataFrame): DataFrame returned from clean_data() function
        database_filename (str): File path to save the database into, eg. 'DisasterResponse.db'

    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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