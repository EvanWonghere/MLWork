# -*- coding: utf-8 -*-
# @Time    : 2024/11/18 14:10
# @Author  : EvanWong
# @File    : SoftSVM.py
# @Project : MLWork

from Visual import plot_svc_decision_function
import matplotlib.pyplot as plt

import numpy as np
from sklearn.svm import SVC
from sklearn.datasets._samples_generator import make_blobs


if __name__ == '__main__':
    # 生成训练数据
    X, y = make_blobs(n_samples=100, centers=2,
                      random_state=0, cluster_std=0.8)

    # 训练模型且展示结果
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    Clist = [10, 0.1]
    for i in np.arange(2):
        model = SVC(kernel="linear", C=Clist[i]).fit(X, y)
        ax[i].scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="summer")
        plot_svc_decision_function(model, ax[i])
        ax[i].set_title('C = {0:.1f}'.format(Clist[i]), size=14)
    plt.show()
