import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

panda_data = pd.read_csv("supermarket_600.csv")
# feature_name = ["Fresh", "Milk", "Frozen", "Grocery"]
# panda_data = panda_data.tail(10000)
# print(panda_data)
feature_name = ["distance_shop_1",
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
X = panda_data[feature_name]
Y = panda_data["shops_used"]


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)


# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
#
# rf = SVC()
#
# rf.fit(X_train, y_train)
#
#
# x = panda_data[feature_name]
# y = rf.predict(x)
# print(len(x),len(y))
#
# print('score (training): {:.3f}'
#       .format(rf.score(X_train, y_train)))
# print('score (test): {:.3f}'
#       .format(rf.score(X_test, y_test)))




npX_train = X_train.values
npX_test = X_test.values
npY_train = y_train.values
npY_test = y_test.values

print(npY_test)
print(npY_train)

print(len(y_train))

print(len(y_test))

np.savez_compressed('data/fake_data', X_train = npX_train, Y_train = npY_train, X_test = npX_test, Y_test = npY_test)
# print(npX_train)

