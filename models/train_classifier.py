import sys
import pandas as pd
import re
import pickle


from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import words


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')


def load_data(database_filepath):
    """
    Load data from an SQLite database and prepare it for machine learning tasks.

    Args:
    database_filepath (str): Filepath for the SQLite database containing the dataset.

    Returns:
    X (pd.Series): Features extracted from the 'message' column.
    Y (pd.DataFrame): Target labels, excluding unnecessary columns.
    category_names (list): List of target category names.
    """
    # Create a connection engine to the SQLite database
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # Read the table 'disaster_table' from the database
    with engine.connect() as conn:
        df = pd.read_sql_table('disaster_table', conn)

    # Extract the feature (messages)
    X = df['message']
    
    # Drop irrelevant columns for target variables
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    
    # Display the target column names
    print(f'target names = {Y.columns}')
    
    # Get list of category names from the target DataFrame
    category_names = Y.columns.tolist()

    return X, Y, category_names



def tokenize(text, 
             lemmatizer=WordNetLemmatizer(), 
             stop_words=stopwords.words("english"), 
             valid_words=set(words.words())):
    """
    Tokenize and clean input text for NLP tasks.

    This function performs the following steps:
    1. Removes URLs.
    2. Normalizes text to lowercase and removes non-alphabetic characters.
    3. Tokenizes the text into individual words.
    4. Lemmatizes tokens with different parts of speech (noun, verb, adjective, etc.).
    5. Filters tokens by word length and valid words dictionary.
    6. Removes stop words from the final token list.

    Args:
    text (str): Input string to be tokenized.
    lemmatizer (WordNetLemmatizer): Lemmatizer to reduce words to their base form.
    stop_words (list): List of stop words to exclude from tokens.
    valid_words (set): Set of valid English words to keep in tokens.

    Returns:
    clean_tokens (list): List of processed, cleaned tokens.
    """
    # Regex to identify and remove URLs
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Remove URLs from the text
    text = re.sub(url_regex, ' ', text)
    
    # Remove non-alphabetic characters and normalize to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Tokenize the cleaned, lowercase text into words
    tokens = word_tokenize(text.lower(), language='english')
    
    # Additional cleaning of tokens: strip whitespace and normalize case
    tokens = [w.lower().strip() for w in tokens]
    
    # Lemmatize tokens with different parts of speech: noun, verb, adjective, etc.
    tokens = [lemmatizer.lemmatize(w, pos='n') for w in tokens]
    tokens = [lemmatizer.lemmatize(w, pos='v') for w in tokens]
    tokens = [lemmatizer.lemmatize(w, pos='a') for w in tokens]
    tokens = [lemmatizer.lemmatize(w, pos='r') for w in tokens]
    tokens = [lemmatizer.lemmatize(w, pos='s') for w in tokens]
    
    # Filter out tokens that are too short (< 3 characters) or too long (> 10 characters)
    tokens = [w for w in tokens if len(w) > 2]
    tokens = [w for w in tokens if len(w) <= 10]
    
    # Keep only tokens found in the valid words set
    tokens = [w for w in tokens if w.lower() in valid_words]
    
    # Remove stop words from the token list
    clean_tokens = [w for w in tokens if w not in stop_words]
    
    return clean_tokens



def build_model():
    """
    Build a machine learning pipeline for multi-output classification.

    This pipeline consists of the following steps:
    1. Tokenization and vectorization using CountVectorizer.
    2. Term Frequency-Inverse Document Frequency (TF-IDF) transformation.
    3. Multi-output classification using a DecisionTreeClassifier with specified hyperparameters.

    Returns:
    pipeline (Pipeline): A scikit-learn Pipeline object that preprocesses the data and trains a model.
    """
    # Define the hyperparameters for the DecisionTreeClassifier
    parameters = {
        'max_depth': 15,                # Maximum depth of the tree
        'min_samples_split': 2,         # Minimum number of samples required to split an internal node
        'min_samples_leaf': 5,          # Minimum number of samples required to be at a leaf node
        'random_state': 42,             # Seed for reproducibility
        'class_weight': 'balanced'      # Adjust class weights to handle class imbalance
    }
    
    # Create a pipeline that preprocesses the data and applies a multi-output classifier
    pipeline = Pipeline([
        # Step 1: Tokenize and vectorize the text data
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None, max_features=500)),
        
        # Step 2: Apply TF-IDF transformation to normalize word frequencies
        ('tfidf', TfidfTransformer()),
        
        # Step 3: Use MultiOutputClassifier with DecisionTreeClassifier as the base model
        ('mclf', MultiOutputClassifier(DecisionTreeClassifier(**parameters)))
    ], verbose=True)
    
    return pipeline


def build_model_CV():
    """
    Build a machine learning pipeline with hyperparameter tuning using GridSearchCV.

    This function creates a pipeline for multi-output classification, including:
    1. Tokenization and vectorization using CountVectorizer.
    2. TF-IDF transformation.
    3. Multi-output classification using a DecisionTreeClassifier.
    
    The function uses GridSearchCV to find the best hyperparameters for the classifier.

    Returns:
    cv (GridSearchCV): A GridSearchCV object that optimizes the model's hyperparameters.
    """
    # Create a pipeline that preprocesses the data and applies a multi-output classifier
    pipeline = Pipeline([
        # Step 1: Tokenize and vectorize the text data
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None, max_features=500)),
        
        # Step 2: Apply TF-IDF transformation to normalize word frequencies
        ('tfidf', TfidfTransformer()),
        
        # Step 3: Use MultiOutputClassifier with DecisionTreeClassifier as the base model
        ('mclf', MultiOutputClassifier(DecisionTreeClassifier(random_state=42, class_weight='balanced')))
    ])

    # Define hyperparameter grid for GridSearchCV
    parameters = {
        'mclf__estimator__max_depth': [15, 25, 75],          # Maximum depth of the tree
        'mclf__estimator__min_samples_split': [2, 5, 10],    # Minimum samples required to split an internal node
        'mclf__estimator__min_samples_leaf': [5, 10, 20],     # Minimum samples required to be at a leaf node
    }
    
    # Create a GridSearchCV object to optimize hyperparameters
    cv = GridSearchCV(pipeline, parameters, verbose=3, n_jobs=-1, cv=3)

    return cv




def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of a trained machine learning model on the test dataset.

    This function calculates various classification metrics including accuracy, precision, recall, and F1-score for each category.

    Args:
    model (object): The trained model to be evaluated.
    X_test (pd.DataFrame): Features of the test dataset.
    Y_test (pd.DataFrame): True labels of the test dataset.
    category_names (list): List of category names corresponding to the labels in Y_test.

    Returns:
    results_df (pd.DataFrame): A DataFrame containing accuracy, precision, recall, and F1-score for each category.
    """
    # Get the category names from the Y_test DataFrame
    category_names = Y_test.columns.tolist()
    
    # Predict the labels for the test features
    y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)

    # Initialize lists to store evaluation metrics
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # Loop through each category to compute evaluation metrics
    for column in category_names:
        y_true = Y_test[column]
        print(f'****{column}*******')

        # Calculate and print the classification report
        print(classification_report(y_true, y_pred[column]))

        # Calculate accuracy
        accuracy = (y_pred[column].values == y_true.values).mean()

        # Calculate precision, recall, and F1-score for multiclass classification
        precision = precision_score(y_true, y_pred[column], average='weighted')  # or 'micro', 'weighted'
        recall = recall_score(y_true, y_pred[column], average='weighted')        # or 'micro', 'weighted'
        f1 = f1_score(y_true, y_pred[column], average='weighted')                # or 'micro', 'weighted'

        # Append the metrics to their respective lists
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # Print accuracy for the current category
        print(f'accuracy = {accuracy}\n\n')

    # Create a DataFrame with the results for each category
    results_df = pd.DataFrame({
        'accuracy': accuracies,
        'precision': precisions,
        'recall': recalls,
        'f1_score': f1_scores
    }, index=category_names)

    # Print the results DataFrame
    print(results_df)

    return results_df


def save_model(model, model_filepath):
    """
    Save a trained machine learning model to a specified file using pickle.

    This function serializes the model object and writes it to a file, allowing for easy retrieval later.

    Args:
    model (object): The trained machine learning model to be saved.
    model_filepath (str): The file path where the model will be saved, including the file name and extension.
    """
    # Serialize the model and write it to the specified file
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    """
    Main entry point for the disaster response classifier training script.

    This function orchestrates the following steps:
    1. Loads the data from a specified database.
    2. Splits the data into training and testing sets.
    3. Builds and trains a machine learning model.
    4. Evaluates the model's performance.
    5. Saves the trained model to a specified file.
    
    The function expects two command-line arguments: 
    the database filepath and the model filepath.
    """
    if len(sys.argv) == 3:
        # Unpack command-line arguments for database and model file paths
        database_filepath, model_filepath = '../data/DisasterResponse.db', '../models/classifier.pkl'

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        # Split the data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
        
        print(f'X.shape = {X.shape}')
        print(f'Y.shape = {Y.shape}')
        print(f'X_train.shape = {X_train.shape}')
        print(f'Y_train.shape = {Y_train.shape}')

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        try:
            # Print the best hyperparameters if available
            print(f'model.best_params_ = {model.best_params_}')
        except:
            # Fallback to print model parameters if no best params are available
            print(f'model.get_params = {model.get_params}')

        # Vectorize the training data
        vect = CountVectorizer(tokenizer=tokenize, token_pattern=None, max_features=500)
        vector = vect.fit_transform(X_train)
        print(f'vect.vocabulary_ = {vect.vocabulary_.keys()}')
        print(f'number of features created = {len(vect.vocabulary_.keys())}')

        print('Evaluating model...')
        # Evaluate the model and save results to CSV
        results_df = evaluate_model(model, X_test, Y_test, category_names)
        results_df.to_csv('../models/metrics_results.csv')

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')




if __name__ == '__main__':
    main()

