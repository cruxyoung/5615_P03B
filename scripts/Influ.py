import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from rbf_test import rbf_svm_influence
from rbf_test_fig import generate_fig


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
        rbf_svm_influence(test_idx=20)




    def visualization(self):
        generate_fig()


test = Influ()
# test.load_data('../source_datasets/supermarket_600.csv')
# test.load_data('../source_datasets/fruit_data.txt')


# label = "shops_used"
# test.convert(feature, label=label)
test.cal_influe()
# test.visualization()
# print(test.dataset)