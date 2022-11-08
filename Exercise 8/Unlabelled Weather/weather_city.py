#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 18:06:21 2022

@author: MichaelKuby/Professor Greg Baker

Much of the code for this program was inspired by professor Baker's code
from colour_predict_hint.py given to us in exercise 7 from CMPT 353 fall
semester at Simon Fraser University of 2022.
"""

import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

OUTPUT_TEMPLATE = (
    'Bayesian classifier:     {bayes_t:.3f}  {bayes:.3f}\n'
    'kNN classifier:          {bayes_t:.3f}  {kNN:.3f}\n'
    'Rand forest classifier:  {rand_forests_t:.3f}  {rand_forests:.3f}\n'
    'Gradient Boosting Classifier: {GBC_model_t:.3f}  {GBC_model:.3f}'
)

def get_data(filename):
    data = pd.read_csv(filename)
    return data

def main():
    labelled = get_data(sys.argv[1])
    unlabelled = get_data(sys.argv[2])
    to_predict = unlabelled.drop('city', axis=1)
    to_predict = pd.DataFrame(to_predict).values
    
    X = labelled.drop(['city'], axis=1).values
    y = labelled['city'].values
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    
    # TODO: create some models
    # Bayes GaussianNB
    bayes_model = make_pipeline(
        StandardScaler(),
        GaussianNB()
        )
    
    #bayes_hsv_model
    kNN_model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors = 15)
        )
    
    # Random Forests rbg model
    rand_forests_model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, max_depth=18)
        )
    
    # Gradient Boosting Classifier
    GBC_model = make_pipeline(
        StandardScaler(),
        GradientBoostingClassifier(n_estimators=50, max_depth=2, min_samples_leaf=0.1)
        )
    
    # train each model and output image of predictions
    models = [bayes_model, kNN_model, rand_forests_model, GBC_model]
    for i, m in enumerate(models):
        m.fit(X_train, y_train)
    
    """
    print(OUTPUT_TEMPLATE.format(
        bayes_t=bayes_model.score(X_train, y_train),
        bayes=bayes_model.score(X_valid, y_valid),
        kNN_t=kNN_model.score(X_train, y_train),
        kNN=kNN_model.score(X_valid, y_valid),
        rand_forests_t=rand_forests_model.score(X_train, y_train),
        rand_forests=rand_forests_model.score(X_valid, y_valid),
        GBC_model_t=GBC_model.score(X_train, y_train),
        GBC_model=GBC_model.score(X_valid, y_valid)))
    """
    
    predictions = rand_forests_model.predict(to_predict)
    
    pd.Series(predictions).to_csv(sys.argv[3], index=False, header=False)
    
    print(rand_forests_model.score(X_valid, y_valid))
    
    df = pd.DataFrame({'truth': y_valid, 'prediction': rand_forests_model.predict(X_valid)})
    # print(df[df['truth'] != df['prediction']])
    
if __name__ == '__main__':
    main()