#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 12:33:00 2022

@author: MichaelKuby
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X,y)

export_graphviz(
    tree_clf, out_file=("iris_tree.dot"), 
    feature_names=iris.feature_names[2:], 
    class_names=iris.target_names, 
    rounded=True, 
    filled=True )