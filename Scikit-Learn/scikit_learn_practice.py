# encoding=utf-8

# 实现决策树

import numpy as np
import pandas as pd


def main():
    # Pre-processing 预处理
    from sklearn.datasets import load_iris
    iris = load_iris()
    # print(iris)
    print(len(iris["data"]))

    from sklearn.cross_validation import train_test_split
    # test_size 就是比率，验证数据集占整个的百分之20
    train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target,
                                                                        test_size=0.2, random_state=1)

    # Modeling 建模
    from sklearn import tree
    # 使用分类器，而不是回归器
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    # fit 就是建立了决策树的关系
    clf.fit(train_data, train_target)
    y_pred = clf.predict(test_data)

    # Verify 验证
    from sklearn import metrics
    print(metrics.accuracy_score(y_true=test_target, y_pred=y_pred))
    # 混淆矩阵
    print(metrics.confusion_matrix(y_true=test_target, y_pred=y_pred))

    with open("./tree.dot", "w") as fw:
        tree.export_graphviz(clf, out_file=fw)


if __name__ == "__main__":
    main()
