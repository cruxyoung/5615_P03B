# encoding=utf-8

import numpy as np
import pandas as pd


def main():
    # 数据结构 Data Structure
    s = pd.Series([i*2 for i in range(1, 11)])
    print(type(s))
    dates = pd.date_range("20170301", periods=8)
    df = pd.DataFrame(np.random.rand(8, 5), index=dates, columns=list("ABCDE"))
    print(df)
    # df2 = pd.DataFrame({"A": 1, "E": pd.Timestamp("20170301"),
    #                     "C": pd.Series(1, index=list(range(4)), dtype="float32"),
    #                     "D": np.array([3]*4, dtype="float32"), "E": pd.Categorical(["pol", "stu", "tear", "doc"])})
    # print(df2)

    # 基本操作
    # print(df.head(3))
    # print(df.tail(3))
    # print(df.index)
    # print(df.values)
    # print(df.T)

    print(df.sort_index(axis=1, ascending=False))
    # describe 数量、平均值、最小值、最小4分为、中位数、上4分位数，最大值
    print(df.describe())
    print(df["A"])
    print(df[:3])
    # 用index进行切片
    print(df["20170301": "20170304"])
    print(df.loc[dates[0]])
    print(df.loc["20170301": "20170304", ["B", "D"]])
    print(df.at[dates[0], "C"])

    # 通过下标进行选择
    print(df.iloc[1:3, 2:4])
    print(df.iloc[1, 4])
    print(df.iat[1, 4])

    print(df[df.B > 0][df.A < 0])
    print(df[df > 0])


if __name__ == "__main__":
    main()
