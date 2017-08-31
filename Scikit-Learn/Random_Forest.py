# encoding=utf-8
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


def main():

    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)

    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=2, max_features='auto', max_leaf_nodes=None,
                           min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                           oob_score=False, random_state=0, verbose=0, warm_start=False)

    print(clf.feature_importances_)

    print(clf.predict([0, 0, 0, 0]))

    #
    # iris = load_iris()
    # df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df = pd.read_csv('WholesData.csv')
    print(df.head())

    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
    print(df.head())

    train, test = df[df['is_train'] == True], df[df['is_train'] == False]
    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    print('Start Processing!')
    features = df[:4]

    print(features)


if __name__ == "__main__":
    main()

    # max_depth 树的最大深度，如果是none则会扩展到叶子节点至少包括min_samples_split
    # max_feature 用于被区分的特征树
    # max_leaf_nodes 在第一个最好的形式中用这个节点去增长树，int or None, optional (Default=0.)
    # min_impurity_split 在树增长时最初停止的门槛，节点会拆分如果纯度到达了这个门槛，除非这个节点已经是叶子节点了
    # min_samples_leaf 最小数量的被提供的样本实例成为叶子节点，如果是int就会将这个作为数量，如果是float就会看作是百分比
    # min_samples_split 样本的内部节点的最小数量
    # min_weight_fraction_leaf 在叶子节点所需输入的样本的权重总和的最小加权分数
    # n_estimators 子-适合的估计器的集合
    # n_jobs 为fit和predict的并行运行的工作数量，如果等于-1，则将工作数设置称核心数
    # oob_score 是否需要用 out-of-bag samples 来评估泛化准确性
    # random_state 随机种子
    # verbose 控制树建立过程中的冗余
    # warm_start 当设置为True时，重新使用上一次调用的解决方案来适应并向集合添加更多估计量，否则只适合整个森林