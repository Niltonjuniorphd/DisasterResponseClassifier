# %%
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


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    with engine.connect() as conn:
        df = pd.read_sql_table('disaster_table', conn)

    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    #Y = Y.iloc[:,0:3]
    print(Y.columns)

    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize_old(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        # print(clean_tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


def tokenize(text, lemmatizer = WordNetLemmatizer(), stop_words = stopwords.words("english")):
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex,' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    tokens = [lemmatizer.lemmatize(w, pos='v') for w in tokens]

    clean_tokens = [w.strip() for w in tokens]
    
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
        ('tfidf', TfidfTransformer()),
        ('mclf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])

    parameters = {
        'mclf__estimator__n_estimators': [100, 200],
        #'mclf__estimator__max_depth': [10, 50],
        'mclf__estimator__min_samples_split': [2, 10],
        #'mclf__estimator__min_samples_leaf': [1, 4],
        #'mclf__estimator__max_features': ['sqrt', 'log2'],
        #'mclf__estimator__bootstrap': [True, False]
    }

    cv = GridSearchCV(pipeline, parameters, verbose=3, n_jobs=-1, cv=3)

    return cv


def build_model_SVC():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('scaler', StandardScaler()),
        ('mclf', MultiOutputClassifier(SVC(random_state=42)))
    ])

    parameters = {
        'scaler__with_mean': [True, False],
        'mclf__estimator__kernel': ['linear', 'rbf'],
        'mclf__estimator__C':[1, 10]
    }

    cv = GridSearchCV(pipeline, parameters, verbose=3, n_jobs=1, cv=3)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)

    for column in category_names:
        y_true = Y_test[column]
        print(f'****{column}*******')
        print(classification_report(y_true, y_pred[column]))
        accuracy = (y_pred[column].values == y_true.values).mean()
        print(f'accuracy_form_compare_mean() = {accuracy}')

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        # database_filepath, model_filepath = sys.argv[1:]
        database_filepath, model_filepath = '../data/DisasterResponse.db', '../models/classifier.pkl'

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=42)
        print(f'X.shape = {X.shape}')
        print(f'Y.shape = {Y.shape}')
        print(f'X_train.shape = {X_train.shape}')
        print(f'Y_train.shape = {Y_train.shape}')

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


# %%
if __name__ == '__main__':
    main()
# %%
