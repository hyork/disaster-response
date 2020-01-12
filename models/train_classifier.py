import sys

# import libraries
import nltk
nltk.download(['punkt', 'wordnet'])

import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    """ 
    Load data from a SQLite database
    
    Parameters: 
    database_filepath (str): File path of the SQLite database, eg. 'DisasterResponse.db'
  
    Returns: 
    X (DataFrame): DataFrame containing the messages
    Y (DataFrame): DataFrame containing the categories
    categories (array): Array of category names
    
    """
    engine = 'sqlite:///{}'.format(database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'])
    categories = Y.columns.values
    return X, Y, categories


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ 
    Build a supervised multi-label classification model
  
    Returns: 
    Classification Pipeline
    
    """
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', OneVsRestClassifier(LinearSVC())),
                    ])
    # defining parameters for tuning 
    parameters = {  'clf__estimator__C': [0.1, 1, 10, 100, 1000],  
                    'clf__estimator__dual':[True,False]
                 }
    cv = GridSearchCV(pipeline, parameters, refit = True, verbose = 3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ 
    Print accuracy score, precision, recall, and f1 score for each category
    
    Parameters: 
    model : Model returned from build_model()
    X_test (Array): Feature test set 
    Y_test (Array) : Category test set
    category_names (Array): Category names
    
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    print(accuracy_score(Y_test, y_pred))


def save_model(model, model_filepath):
    """ 
    Save model into a pickle file
    
    Parameters: 
    model: Model returned from build_model() function
    model_filepath (str): File name of the pickle file, eg. 'classifier.pkl'
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()