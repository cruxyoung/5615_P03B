from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import math
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model as linear_model
import sklearn.preprocessing as preprocessing
import scipy
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse

sns.set(color_codes=True)

from .load_animals import load_animals
from ..influence.image_utils import plot_flat_colorimage, plot_flat_colorgrad


def generate_fig():
    num_classes = 2
    # num_train_ex_per_class = 330
    # num_test_ex_per_class = 110

    num_train_ex_per_class = 180
    num_test_ex_per_class = 60

    model_name = 'dogfish_%s_%s' % (num_train_ex_per_class, num_test_ex_per_class)
    image_data_sets = load_animals(
        num_train_ex_per_class=num_train_ex_per_class,
        num_test_ex_per_class=num_test_ex_per_class,
        classes=['dog', 'fish'])



    X_train = image_data_sets.train.x
    X_test = image_data_sets.test.x
    Y_train = image_data_sets.train.labels * 2 - 1
    Y_test = image_data_sets.test.labels * 2 - 1

    f = np.load('output/rbf_results.npz')

    test_idx = f['test_idx']
    distances = f['distances']
    flipped_idx = f['flipped_idx']
    rbf_margins_test = f['rbf_margins_test']
    rbf_margins_train = f['rbf_margins_train']
    # inception_Y_pred_correct = f['inception_Y_pred_correct']
    rbf_predicted_loss_diffs = f['rbf_predicted_loss_diffs']*1
    # rbf_predicted_loss_diffs = f['rbf_predicted_loss_diffs']

    # inception_predicted_loss_diffs = f['inception_predicted_loss_diffs']


    sns.set_style('white')
    fontsize=14

    fig, axs = plt.subplots(1, sharex=True, sharey=False, figsize=(6, 3))

    num_train = len(flipped_idx)
    color_vec = np.array(['g'] * num_train)
    color_vec[flipped_idx] = 'r'
    color_vec = list(color_vec)

    axs.scatter(distances, rbf_predicted_loss_diffs, color=color_vec)
    axs.set_ylim(-0.03, 0.03)
    # c = pow(10,-1000000000000)
    # print(c)
    # axs[0].set_ylim(-0.03*c, 0.03*c)

    axs.set_yticks((-0.03, 0, 0.03))
    axs.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    axs.set_xlabel('Euclidean distance', fontsize=fontsize)
    axs.set_ylabel('$-\mathcal{I}_\mathrm{up, loss} \ /\ n$', fontsize=fontsize)

    # axs[1].scatter(distances, inception_predicted_loss_diffs, color=color_vec)
    # axs[1].set_ylim(-0.0005, 0.0005)
    # axs[1].set_yticks((-0.0005, 0, 0.0005))
    # axs[1].ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    # axs[1].set_xlabel('Euclidean distance', fontsize=fontsize)

    plt.tight_layout()
    plt.show()
    # plot_flat_colorimage((X_test[test_idx, :] + 1) / 2, 0, side=299)

if __name__ == '__main__':
    generate_fig()