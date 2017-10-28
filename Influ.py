import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .scripts.rbf_test import rbf_svm_influence
from .scripts.rbf_test_fig import generate_fig


class Influ(object):

    def __init__(self):
        self.dataset = None

    # load data from the specified file
    def load_data(self, filename):
        if '.csv' in filename:
            self.dataset = pd.read_csv(filename)
        elif '.txt' in filename:
            self.dataset =  pd.read_table(filename)
        elif '.xlsx' in filename:
            self.dataset = pd.read_excel(filename)
        else:
            return 'file type not supported'

    # convert the data into correct format
    def convert(self, feature, label):
        X = self.dataset[feature]
        Y = self.dataset[label]
        # split training, testing data
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

        npX_train = X_train.values
        npX_test = X_test.values
        npY_train = y_train.values
        npY_test = y_test.values

        np.savez_compressed('fake_data', X_train=npX_train, Y_train=npY_train, X_test=npX_test, Y_test=npY_test)
        
    # train the compute influence
    def cal_influe(self, test_idx=None, gamma=None):
        rbf_svm_influence(test_idx=test_idx,gamma=gamma)



    # change scale for different view
    def visualization(self, scale = None):
        generate_fig(scale = scale)

if __name__ == '__main__':
    test = Influ()
    # test.load_data('supermarket_600.csv')
    # # test.load_data('../source_datasets/fruit_data.txt')
    # print(test.dataset)

    # label = "shops_used"
    # test.convert("customer_id", label=label)
    test.cal_influe(50)
    # test.visualization()
    # print(test.dataset)

