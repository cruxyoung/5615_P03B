from test.Influ import Influ
import unittest
from pandas import DataFrame
import os.path


class TestDataFetch(unittest.TestCase):
    # test the load data function
    def test_load_data(self):
        test = Influ()
        test.load_data('supermarket_600.csv')
        self.assertTrue(type(test.dataset), type(DataFrame()))

    def test_convert(self):
        test = Influ()
        test.load_data('supermarket_600.csv')
        
        label = "shops_used"
        test.convert("customer_id", label=label)
        self.assertTrue(os.path.exists("fake_data.npz"))
        if os.path.exists("fake_data.npz"):
            os.remove("fake_data.npz")

    def test_cal_influe(self):
        test = Influ()
        test.load_data('supermarket_600.csv')
        feature = ["distance_shop_1",
                "distance_shop_2",
                "distance_shop_3",
                "distance_shop_4",
                "distance_shop_5",
                "products_purchased_shop_1",
                "products_purchased_shop_2",
                "products_purchased_shop_3",
                "products_purchased_shop_4",
                "products_purchased_shop_5"
                ]
        label = "shops_used"
        test.convert(feature, label)
        test.cal_influe(10)


    def test_visualization(self):
        test = Influ()
        test.load_data('supermarket_600.csv')
        feature = ["distance_shop_1",
                "distance_shop_2",
                "distance_shop_3",
                "distance_shop_4",
                "distance_shop_5",
                "products_purchased_shop_1",
                "products_purchased_shop_2",
                "products_purchased_shop_3",
                "products_purchased_shop_4",
                "products_purchased_shop_5"
                ]
        label = "shops_used"
        test.convert(feature, label)
        test.cal_influe(10)
        test.visualization(0.05)
        if os.path.exists("fake_data.npz"):
            os.remove("fake_data.npz")



if __name__ == '__main__':
    unittest.main()