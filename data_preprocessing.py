import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from sklearn.utils import class_weight

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def gendata(doPCA=False):
    data = pd.read_csv('LengthOfStay.csv')
    #Save the length of stay in a different variable
    labels = data['lengthofstay']
    # Drop columns that we dont need like specific dates, or the id of the patient
    data = data.drop(["eid", "vdate", "discharged", "lengthofstay"], axis=1)
    # Add dummy encoding for the object and type variables
    # For example, turn gender column into 2 columns, where a male will be 1 in the first column
    # and a 0 in the second column, and a female will be the inverse
    data = pd.get_dummies(data, columns=['rcount'])
    data = pd.get_dummies(data, columns=['gender'])
    data = pd.get_dummies(data, columns=['facid'])

    if not doPCA:
        hematocrit = data[['hematocrit']].values
        data['hematocrit'] = preprocessing.StandardScaler().fit_transform(hematocrit)

        bloodureanitro = data[['neutrophils']].values
        data['neutrophils'] = preprocessing.RobustScaler().fit_transform(bloodureanitro)

        sodium = data[['sodium']].values
        data['sodium'] = preprocessing.StandardScaler().fit_transform(sodium)

        glucose = data[['glucose']].values
        data['glucose'] = preprocessing.StandardScaler().fit_transform(glucose)

        bloodureanitro = data[['bloodureanitro']].values
        data['bloodureanitro'] = preprocessing.RobustScaler().fit_transform(bloodureanitro)

        creatinine = data[['creatinine']].values
        data['creatinine'] = preprocessing.StandardScaler().fit_transform(creatinine)

        bmi = data[['bmi']].values
        data['bmi'] = preprocessing.StandardScaler().fit_transform(bmi)

        pulse = data[['pulse']].values
        data['pulse'] = preprocessing.StandardScaler().fit_transform(pulse)

        respiration = data[['respiration']].values
        data['respiration'] = preprocessing.StandardScaler().fit_transform(respiration)
        
    # Seperate for train and test
    train_X = data.head(n=80000).to_numpy()
    test_X = data.tail(n=20000).to_numpy()
    train_Y = labels.head(n=80000).to_numpy()
    test_Y = labels.tail(n=20000).to_numpy()

    return train_X, test_X, train_Y, test_Y

