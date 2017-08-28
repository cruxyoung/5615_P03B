from data_examining.data_fetch import fetch_data
import unittest
from pandas import DataFrame

class TestDataFetch(unittest.TestCase):
    def test_fetch_data_csv(self):
        a=fetch_data('../source_datasets/supermarket.csv')
        # print(type(a))
        # print(type(DataFrame()))
        self.assertTrue(type(a), type(DataFrame()))

    def test_fetch_data_txt(self):
        a=fetch_data('../source_datasets/fruit_data.txt')
        # print(type(a))
        # print(type(DataFrame()))
        self.assertTrue(type(a), type(DataFrame()))

    def test_fetch_data_xlsx(self):
        a=fetch_data('../source_datasets/Online Retail.xlsx')
        # print(type(a))
        # print(type(DataFrame()))
        self.assertTrue(type(a), type(DataFrame()))




if __name__ == '__main__':
    unittest.main()