from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import math
import copy
import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model
import sklearn.preprocessing as preprocessing
import scipy
import scipy.linalg as slin
from .load_animals import load_animals

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

from sklearn.metrics.pairwise import rbf_kernel

from ..influence.smooth_hinge import SmoothHinge
# import influence.dataset as dataset
from ..influence import dataset as dataset
from ..influence.dataset import DataSet

def rbf_svm_influence(gamma = None,
                      test_idx = None):

    num_classes = 2
    # num_train_ex_per_class = 781
    # num_test_ex_per_class = 261

    num_train_ex_per_class = 450
    num_test_ex_per_class = 150

    # dataset_name = 'dogfish_%s_%s' % (num_train_ex_per_class, num_test_ex_per_class)
    image_data_sets = load_animals(
        num_train_ex_per_class=num_train_ex_per_class,
        num_test_ex_per_class=num_test_ex_per_class,
        classes=['dog', 'fish'])

    ### Generate kernelized feature vectors
    X_train = image_data_sets.train.x
    X_test = image_data_sets.test.x

    Y_train = (np.copy(image_data_sets.train.labels)-1) * 2 - 1
    Y_test = (np.copy(image_data_sets.test.labels)-1) * 2 - 1



    num_train = X_train.shape[0]
    num_test = X_test.shape[0]

    X_stacked = np.vstack((X_train, X_test))
    # gamma need to be changed min = 0.00002
    # 0.000013 current best for fdata
    # 0.00078best for fakedata
    # this value needs to be changed for different type of data
    if gamma is None:
        gamma = 0.00078
    weight_decay = 0.0001

    K = rbf_kernel(X_stacked, gamma = gamma / num_train)
    # K = linear_kernel(X_stacked)

    L = slin.cholesky(K, lower=True)
    L_train = L[:num_train, :num_train]
    L_test = L[num_train:, :num_train]
    ### Compare top 5 influential examples from each network
    # choose the training dataset as standard
    if test_idx is None:
        test_idx = 50

    ## RBF
    # weight_decay = 0.001
    input_channels = 1
    weight_decay = 0.001
    batch_size = num_train
    initial_learning_rate = 0.001
    keep_probs = None
    max_lbfgs_iter = 1000
    use_bias = False
    decay_epochs = [1000, 10000]

    tf.reset_default_graph()

    X_train = image_data_sets.train.x
    # change value of label
    Y_train = (image_data_sets.train.labels-1) * 2 - 1
    train = DataSet(L_train, Y_train)
    test = DataSet(L_test, Y_test)


    data_sets = base.Datasets(train=train, validation=None, test=test)
    input_dim = data_sets.train.x.shape[1]

    # Train with hinge
    rbf_model = SmoothHinge(
        temp=0,
        use_bias=use_bias,
        input_dim=input_dim,
        weight_decay=weight_decay,
        num_classes=num_classes,
        batch_size=batch_size,
        data_sets=data_sets,
        initial_learning_rate=initial_learning_rate,
        keep_probs=keep_probs,
        decay_epochs=decay_epochs,
        mini_batch=False,
        train_dir='output',
        log_dir='log',
        model_name='dogfish_rbf_hinge_t-0')

    rbf_model.train()
    hinge_W = rbf_model.sess.run(rbf_model.params)[0]

    # Then load weights into smoothed version
    tf.reset_default_graph()
    rbf_model = SmoothHinge(
        temp=0.001,
        use_bias=use_bias,
        input_dim=input_dim,
        weight_decay=weight_decay,
        num_classes=num_classes,
        batch_size=batch_size,
        data_sets=data_sets,
        initial_learning_rate=initial_learning_rate,
        keep_probs=keep_probs,
        decay_epochs=decay_epochs,
        mini_batch=False,
        train_dir='output',
        log_dir='log',
        model_name='dogfish_rbf_hinge_t-0.001')

    params_feed_dict = {}
    params_feed_dict[rbf_model.W_placeholder] = hinge_W
    rbf_model.sess.run(rbf_model.set_params_op, feed_dict=params_feed_dict)
    # get value of influence function
    rbf_predicted_loss_diffs = rbf_model.get_influence_on_test_loss(
        [test_idx],
        np.arange(len(rbf_model.data_sets.train.labels)),
        force_refresh=True)


    x_test = X_test[test_idx, :]
    y_test = Y_test[test_idx]
    # euclidean distance for visualization
    distances = dataset.find_distances(x_test, X_train)
    # flipped_idx indicates if the label of training is identical to one of chosen test_idx
    flipped_idx = Y_train != y_test

    rbf_margins_test = rbf_model.sess.run(rbf_model.margin, feed_dict=rbf_model.all_test_feed_dict)
    rbf_margins_train = rbf_model.sess.run(rbf_model.margin, feed_dict=rbf_model.all_train_feed_dict)

    # save the data for visualization
    np.savez(
        'output/rbf_results',
        test_idx=test_idx,
        distances=distances,
        flipped_idx=flipped_idx,
        rbf_margins_test=rbf_margins_test,
        rbf_margins_train=rbf_margins_train,
        rbf_predicted_loss_diffs=rbf_predicted_loss_diffs,
    )

    from .rbf_test_fig import generate_fig

    generate_fig()

if __name__ == '__main__':
    rbf_svm_influence()