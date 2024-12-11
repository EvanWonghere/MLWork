# -*- coding: utf-8 -*-
# @Time    : 2024/11/18 14:10
# @Author  : EvanWong
# @File    : KernelSVM.py
# @Project : MLWork


from Visual import plot_svc_decision_function
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.datasets._samples_generator import make_circles


if __name__ == '__main__':
    # Kernel SVM
    # 生成训练数据

    X, y = make_circles(100, factor=.1, noise=.1)
    # 训练模型
    clf = SVC(kernel='rbf', C=1E6)
    clf.fit(X, y)

    # 结果展示
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(clf)
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none')
    plt.show()
