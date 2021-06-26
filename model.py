import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv('hiring.csv')
# print(data.head())
# print(data.info())

data['experience'].fillna(0, inplace = True)
data['test_score'].fillna(data['test_score'].mean(), inplace = True)

def string_to_number(word):
    d = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'zero': 0, 0: 0}
    return d[word]

data['experience'] = data['experience'].apply(lambda x: string_to_number(x))
# print(data.info())

x = data.iloc[:, : -1].values
y = data.iloc[:, -1].values

reg = LinearRegression()
reg.fit(x, y)

# Saving Model
pickle.dump(reg, open('model.pkl', 'wb'))

# Loading Model
model = pickle.load(open('model.pkl', 'rb'))

print(round(model.predict([[9, 9, 9]])[0], 2))