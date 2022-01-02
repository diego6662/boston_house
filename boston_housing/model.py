import re
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle


def get_dataset() -> pd.DataFrame:
    df = pd.read_csv('data/housing.csv', names = [
        'crim','zn','indus', 'chas', 'nox', 'rm', 'age', 
        'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv'
    ])
    return df

def save_model(model):
    filename = 'model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def proccessing_pipeline():
    pipeline = Pipeline(
        [
            ('StandardScaler', StandardScaler()),
            ('SVR',SVR())
        ]
    )
    
    params_grid = {
        'SVR__C': [0.01,0.1,1,10,100],
    }
    model = GridSearchCV(pipeline, params_grid)
    return model


def build_model(training=True):
    if training:
        boston_df = get_dataset()
        X, y = boston_df.iloc[:,:-1], boston_df.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
        usefull_features = ['zn', 'rm', 'dis']
        model = proccessing_pipeline()
        impute = SimpleImputer(missing_values=0.0, strategy='mean')
        X_train = X_train[usefull_features]
        X_train = impute.fit_transform(X_train)
        model.fit(X_train,y_train)
        save_model(model.best_estimator_)
        predicts = model.predict(X_test[usefull_features]) 
        error = mean_squared_error(y_test, predicts)
        print(error)
    else:
        filename = 'model.pkl'
        with open(filename, 'rb') as file:
            pickle_model = pickle.load(file)
        return pickle_model
