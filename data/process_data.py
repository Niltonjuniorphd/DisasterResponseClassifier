import sys
import pandas as pd
from sqlalchemy import create_engine, inspect

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df

def clean_data(df):
    categories = df['categories'].str.split(pat=';', expand = True)
    category_colnames = categories.loc[0].replace(r'-.*', '', regex=True).tolist()
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = [x.split(sep='-')[1] for x in categories[column]]
        categories[column] = categories[column].astype('str')
    
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    print(f'number of duplicates cleaned: {df.duplicated().sum()}\n')
    df = df.drop_duplicates()
    #df = df.dropna()
    
    return df

def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_table', engine, index=False)  
    
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    print(f'The name of the table in the data base is {table_names}')


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