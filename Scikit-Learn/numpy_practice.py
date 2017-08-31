# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import *


def main():
    # lst = [[1, 3, 5], [2, 4, 6]]
    # print(type(lst))
    # np_lst = np.array(lst)
    # print(type(np_lst))
    # np_lst = np.array(lst, dtype=np.float)
    # print(np_lst.shape)
    # print(np_lst.ndim)
    # print(np_lst.dtype)
    # print(np_lst.itemsize)
    # print(np_lst.size)

    # print(np.zeros([2, 4]))
    # print(np.ones([3, 5]))
    # print("random:")
    # print(np.random.rand(2, 4))
    # print(np.random.rand())

    # print(np.random.randint(1, 10, 3))
    # print(np.random.randn(2, 4))
    # print(np.random.beta(1, 10, 100))

    # Array Opes
    # lst = np.arange(1, 11).reshape([5, -1])
    # print(np.exp(lst))
    # print(np.exp2(lst))
    # print(np.sqrt(lst))
    # print(np.sin(lst))
    # print(np.log(lst))
    # lst = np.array([[[1, 2, 3, 4],
    #                  [4, 5, 6, 7]],
    #                 [[7, 8, 9, 10],
    #                  [10, 11, 12, 13]],
    #                 [[14, 15, 16, 17],
    #                  [18, 19, 20, 21]]
    #                 ])
    # print(lst.max(axis=0))
    # # matrix multiply
    # lst1 = np.array([10, 20, 30, 40])
    # lst2 = np.array([4, 3, 2, 1])
    # print(np.dot(lst1.reshape([2, 2]), lst2.reshape([2, 2])))
    # print(np.concatenate((lst1, lst2), axis=0))
    # print(np.vstack((lst1, lst2)))
    # print(np.split(lst1, 2))

    # liner
    print(np.eye(3))
    print("Matrix:")
    lst = np.array([[1., 2.],
                    [3., 4.]])
    print("\nInv:")
    print(inv(lst))
    print(lst.transpose())
    print(det(lst))
    # eigenvalue特征值
    print(eig(lst))
    y = np.array([[5.], [7.]])
    print(solve(lst, y))

    # Others
    print("FFT:")
    print(np.fft.fft(np.array([1, 1, 1, 1, 1, 1, 1, 1])))
    print(np.corrcoef([1, 0, 1], [0, 2, 1]))

    print("Poly:")
    print(np.poly1d([2, 1, 3]))


if __name__ == "__main__":
    main()
