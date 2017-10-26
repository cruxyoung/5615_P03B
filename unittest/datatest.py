import sys
sys.path.append('..')
from Influ import Influ
import unittest
from pandas import DataFrame

class TestDataFetch(unittest.TestCase):
    # def test_fetch_data_csv(self):
    #     a=fetch_data('../source_datasets/supermarket.csv')
    #     # print(type(a))
    #     # print(type(DataFrame()))
    #     self.assertTrue(type(a), type(DataFrame()))

    def test_load_data(self):
        test = Influ()
        test.load_data('supermarket_600.csv')




if __name__ == '__main__':
    unittest.main()