# -*- coding: utf-8 -*-
# @Time    : 2024/11/18 14:10
# @Author  : EvanWong
# @File    : HardSVM.py
# @Project : MLWork


from Visual import plot_svc_decision_function
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.datasets._samples_generator import make_blobs


if __name__ == '__main__':
    # Hard Margin SVM
    #   生成训练数据
    X, y = make_blobs(n_samples=50, centers=2,
                      random_state=0, cluster_std=0.60)
    #   模型训练
    model = SVC(kernel='linear', C=1E10)
    model.fit(X, y)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
    plot_svc_decision_function(model, plot_support=True)
    plt.show()
