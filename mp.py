import quandl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from statistics import mean
from statistics import mean
from sklearn import preprocessing, svm, cross_validation
from sklearn.model_selection import train_test_split

style.use('fivethirtyeight')

api_key = open('key.txt', 'r').read()


def create_labels(cur_hpi, fu_hpi):
    if fu_hpi > cur_hpi:
        return 1
    else:
        return 0


def moving_average(values):
    return mean(values)


housing_data = pd.read_pickle('HPI.pickle')
housing_data = housing_data.pct_change()
print(housing_data.head())
housing_data.replace([np.inf, -np.inf], np.nan, inplace=True)
housing_data['US_HPI_Future'] = housing_data['United States'].shift(-1)
housing_data.dropna(inplace=True)

housing_data['label'] = list(
    map(
        create_labels, housing_data['United States'], housing_data['US_HPI_Future']
    )
)
print(housing_data.head())
#
# housing_data['ma_apply_example'] = pd.rolling_apply(housing_data['M30'], 10, moving_average)
# print(housing_data.tail())

# Features
X = np.array(housing_data.drop(['label', 'US_HPI_feature'], 1))
X = preprocessing.scale(X)

# Labels
y = np.array(housing_data['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifications = svm.SVC(kernel='linear')
classifications.fit(X_train, y_train)
print(classifications.score(X_test, y_test))
