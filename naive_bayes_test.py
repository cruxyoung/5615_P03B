import numpy
import pandas
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

raw_data = pandas.read_csv('test_data.csv')

data_label = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
result_label = ['Channel']

test_data = raw_data[data_label][:200].as_matrix()
labels = raw_data[result_label][:200].as_matrix().ravel()

check_data = raw_data[data_label][200:300].as_matrix()

# print(test_data)
# print('=======')
# print(labels)

iris = datasets.load_iris()
gnb = GaussianNB().fit(test_data, labels)
print(check_data[0])
print(gnb.predict([check_data[0]]))

