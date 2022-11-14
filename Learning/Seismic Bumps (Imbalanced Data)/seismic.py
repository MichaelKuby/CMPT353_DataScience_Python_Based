import os
from urllib.request import urlopen
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report


# data from https://archive.ics.uci.edu/ml/datasets/seismic-bumps
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff'
DATA_FILE = 'seismic-bumps.arff'


assessment_map = {b'a': 0, b'b': 1, b'c': 2, b'd': 3}
def numeric_features(data):
    data = data.copy()
    data['seismic'] = data['seismic'].map(assessment_map)
    data['seismoacoustic'] = data['seismoacoustic'].map(assessment_map)
    data['ghazard'] = data['ghazard'].map(assessment_map)

    mlb = MultiLabelBinarizer()
    shift = mlb.fit_transform(data['shift'])
    data['shift_0'] = shift[:, 0]
    data['shift_1'] = shift[:, 1]
    data = data.drop('shift', axis=1)
    return data


def train_with(data):
    bump_next_shift = data['class']
    features = data.drop('class', axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(features, bump_next_shift, random_state=4)
    
    model = make_pipeline(
        FunctionTransformer(numeric_features, validate=False),
        SimpleImputer(strategy='mean'),
        GradientBoostingClassifier(n_estimators=50, max_depth=3, min_samples_leaf=0.1)
    )
    model.fit(X_train, y_train)
    print(model.score(X_valid, y_valid))
    print(model.score(X_train, y_train))
    print(classification_report(y_valid, model.predict(X_valid)))
    print("True count in class 1:     ", (y_valid == 1).sum())
    print("Predicted count in class 1:", (model.predict(X_valid) == 1).sum())


def main():
    if not os.path.exists(DATA_FILE):
        with urlopen(DATA_URL) as req, open(DATA_FILE, 'wb') as data:
            data.write(req.read())
    
    data = pd.DataFrame(arff.loadarff(DATA_FILE)[0])
    data['class'] = data['class'].astype(int)
    print(data['class'].value_counts())

    train_with(data)
    
    bump = data[data['class'] == 1]
    nobump = data[data['class'] == 0].sample(n=bump.shape[0])
    balanced_data = bump.append(nobump)
    bump = data[data['class'] == 1]
    nobump = data[data['class'] == 0].sample(n=bump.shape[0], random_state=3)
    balanced_data = bump.append(nobump)
    
    train_with(balanced_data)
    
    

if __name__ == '__main__':
    main()
