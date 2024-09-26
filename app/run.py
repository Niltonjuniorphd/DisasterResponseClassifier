import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine, inspect
import re
from nltk.corpus import stopwords
from nltk.corpus import words


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')


app = Flask(__name__)


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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
inspector = inspect(engine)  # eu
table_names = inspector.get_table_names()  # eu
print(f'table_names => {table_names}')
df = pd.read_sql_table(table_names[0], engine)  # eu
df_metrics = pd.read_csv('../models/metrics_results.csv', index_col=0)
print(f'df_metrics = {df_metrics.index}')
# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre')['message'].count()
    related_counts = df.groupby('related')['message'].count()
    genre_names = list(genre_counts.index)
    related_names = list(related_counts.index)

    df_target = df.drop(['id', 'message', 'original',
                        'genre'], axis=1)
    df_target = df_target.astype(int)

    df_target_means = df_target.mean()
    df_target_names = list(df_target_means.index)

    df_metrics_values = df_metrics['recall'].values
    df_metrics_names = df_metrics.index

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
                    x=related_names,
                    y=related_counts
                )
            ],

            'layout': {
                'title': 'Distribution of the target "Related" (non-binary category)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Related counts"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=df_target_names,
                    y=df_target_means
                )
            ],

            'layout': {
                'title': r'% of value 1 in targets (inbalance)',
                'yaxis': {
                    'title': "%"
                },
                'xaxis': {
                    'title': "Target names"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=df_metrics_names,
                    y=df_metrics_values
                )
            ],

            'layout': {
                'title': r'recall_weighted',
                'yaxis': {
                    'title': "%"
                },
                'xaxis': {
                    'title': "Target names"
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
