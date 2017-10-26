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
        self.assertTrue(type(test.dataset), type(DataFrame()))

    def test_convert(self):
    	test = Influ()
        test.load_data('supermarket_600.csv')
        label = "shops_used"
    	test.convert("customer_id", label=label)
    	



if __name__ == '__main__':
    unittest.main()