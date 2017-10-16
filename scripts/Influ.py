import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Influ(object):

    def __init__(self):
        self.dataset = None

    def load_data(self, filename):
        if '.csv' in filename:
            self.dataset = pd.read_csv(filename)
        elif '.txt' in filename:
            self.dataset =  pd.read_table(filename)
        elif '.xlsx' in filename:
            self.dataset = pd.read_excel(filename)
        else:
            return 'file type not supported'


    def convert(self, feature, label):
        X = self.dataset[feature]
        Y = self.dataset[label]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

        npX_train = X_train.values
        npX_test = X_test.values
        npY_train = y_train.values
        npY_test = y_test.values

        np.savez_compressed('data/fake_data', X_train=npX_train, Y_train=npY_train, X_test=npX_test, Y_test=npY_test)

    def cal_influe(self):
        pass


    def visualization(self):
        pass



test = Influ()
test.load_data('../source_datasets/supermarket.csv')
print(test.dataset)