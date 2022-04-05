import pandas as pd
import os

TRAIN_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../data/train.csv'))
TEST_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../data/test.csv'))

class DataSet():
    def __init__(self, X, y):
        self.X = X
        self.y = y

def get_training_set():
    training_data = pd.read_csv(TRAIN_FILE,sep=",")
    X = training_data.iloc[:, 1:]
    Y = training_data.iloc[:, 0]

    training_set = DataSet(X.values, Y.values)
    return training_set

def get_test_set():
    test_data = pd.read_csv(TEST_FILE, sep=",")
    X = test_data.iloc[:, 1:]
    Y = test_data.iloc[:, 0]

    test_set = DataSet(X.values, Y.values)
    return test_set