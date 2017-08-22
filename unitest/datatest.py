from data_examining.data_fetch import fetch_data
import unittest
from pandas import DataFrame

class TestDataFetch(unittest.TestCase):
    def test_fetch_data(self):
        a=fetch_data('../source_datasets/Supermarket_customer.csv')
        # print(type(a))
        # print(type(DataFrame()))
        self.assertTrue(type(a), type(DataFrame()))


if __name__ == '__main__':
    unittest.main()