import sys
import pandas as pd
from sqlalchemy import create_engine, inspect

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge datasets from provided filepaths.

    This function reads two CSV files: one containing messages and the other containing categories. 
    It merges both datasets on the 'id' column.

    Args:
    messages_filepath (str): Filepath to the CSV file containing messages.
    categories_filepath (str): Filepath to the CSV file containing categories.

    Returns:
    df (pd.DataFrame): A DataFrame containing the merged data from the messages and categories datasets.
    """
    # Load messages dataset from the specified file path
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset from the specified file path
    categories = pd.read_csv(categories_filepath)
    
    # Merge the two datasets on the 'id' column
    df = pd.merge(messages, categories, on="id")
    
    return df


def clean_data(df):
    """
    Clean and preprocess the merged disaster response data.

    This function processes the 'categories' column by splitting it into separate category columns, 
    converting the category values to binary, and removing duplicates from the dataset.

    Args:
    df (pd.DataFrame): The merged DataFrame containing messages and categories.

    Returns:
    df (pd.DataFrame): A cleaned DataFrame with separate binary category columns and no duplicates.
    """
    # Split the 'categories' column into separate category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # Extract column names for categories by removing the numeric part
    category_colnames = categories.loc[0].replace(r'-.*', '', regex=True).tolist()
    categories.columns = category_colnames

    # Convert category values to binary (0 or 1) by extracting the part after the dash
    for column in categories:
        categories[column] = [x.split(sep='-')[1] for x in categories[column]]
        categories[column] = categories[column].astype('str')  # Convert to string for uniformity

    # Drop the original 'categories' column from the dataframe
    df = df.drop('categories', axis=1)
    
    # Concatenate the original dataframe with the new 'categories' binary columns
    df = pd.concat([df, categories], axis=1)
    
    # Print the number of duplicates found and drop duplicates from the dataframe
    print(f'number of duplicates cleaned: {df.duplicated().sum()}\n')
    df = df.drop_duplicates()

    # Check for any remaining duplicates in the dataframe
    assert len(df[df.duplicated()]) == 0

    #removing valued as “2” since those messages do not make sense
    df = df[df['related'] != '2']

    return df


def save_data(df, database_filename):
    """
    Save the cleaned DataFrame to an SQLite database.

    This function saves the processed data to an SQLite database file. 
    It also prints the names of the tables in the database to confirm successful saving.

    Args:
    df (pd.DataFrame): The cleaned DataFrame to be saved.
    database_filename (str): The file path where the SQLite database will be saved.

    """
    # Create a SQLAlchemy engine to connect to the SQLite database
    engine = create_engine(f'sqlite:///{database_filename}')
    
    # Save the DataFrame to a table named 'disaster_table' in the SQLite database
    df.to_sql('disaster_table', engine, index=False, if_exists='replace')  
    
    # Inspect the database to retrieve the names of the tables
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    # Print the name of the saved table to confirm successful operation
    print(f'The name of the table in the database is {table_names}')



def main():
    """
    Main function for loading, cleaning, and saving disaster response data.

    This function:
    1. Loads message and category datasets from CSV files.
    2. Cleans the merged dataset by separating categories and removing duplicates.
    3. Saves the cleaned data to an SQLite database.
    
    The function expects three command-line arguments:
    1. Filepath to the messages CSV file.
    2. Filepath to the categories CSV file.
    3. Filepath to save the cleaned data as an SQLite database.
    """
    if len(sys.argv) == 4:
        # Unpack command-line arguments for file paths
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        
        # Load messages and categories datasets
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        # Clean the merged data
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        # Save the cleaned data to the database
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        # Handle the case where incorrect number of arguments are provided
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')



if __name__ == '__main__':
    main()