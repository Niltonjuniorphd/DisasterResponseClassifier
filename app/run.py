import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine, inspect
import re
from nltk.corpus import stopwords


app = Flask(__name__)


def tokenize_old(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text.lower())
    # print(tokens)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if len(tok) > 3:
            clean_tok = lemmatizer.lemmatize(tok).strip()
            # print(clean_tok)
            clean_tokens.append(clean_tok)
        else:
            pass

    return clean_tokens


def tokenize(text, lemmatizer=WordNetLemmatizer(), stop_words=stopwords.words("english")):
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)

    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    tokens = [lemmatizer.lemmatize(w, pos='v') for w in tokens]

    clean_tokens = [w.strip() for w in tokens]
    clean_tokens = [w for w in tokens if len(w) > 2]

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
                        'genre', 'related'], axis=1)
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
