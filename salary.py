import os
import pandas as pd
import numpy as np

os.chdir("C:\\Users\\user\\Documents\\Python\\Github's\\Deployment-flask-master")

X = pd.read_csv("hiring.csv")

X['experience'].fillna(0, inplace =True)
X['test_score'].fillna(X['test_score'].mean(), inplace =True)

def convert_to_int(word):
    word_dict = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
                 'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'zero':0,0:0}
    return (word_dict[word])


X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))

Df = X.copy()

X.drop(['salary'], axis =1, inplace =True)

Y = X.iloc[:,-1]

from sklearn.linear_model import LinearRegression

M1_Model = LinearRegression().fit(X,Y)

import pickle

pickle.dump(M1_Model,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))