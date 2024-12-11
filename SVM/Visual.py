# -*- coding: utf-8 -*-
# @Time    : 2024/11/18 15:06
# @Author  : EvanWong
# @File    : Visual.py
# @Project : MLWork

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


#   展示结果
def plot_svc_decision_function(model, plot_support=True):
    """Plot the decision function for a 2D SVC"""

    ax = plt.gca()
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(x_lim[0], x_lim[1], 30)
    y = np.linspace(y_lim[0], y_lim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none', edgecolors='black')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
