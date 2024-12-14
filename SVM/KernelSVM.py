# -*- coding: utf-8 -*-
# @Time    : 2024/11/18 14:10
# @Author  : EvanWong
# @File    : KernelSVM.py
# @Project : MLWork
#
# 文件说明：
# 本文件展示如何使用核函数（如 RBF 核）处理线性不可分的数据。
# RBF 核函数通过映射将数据投影到高维特征空间，使原本不可分的数据在高维空间变得线性可分。

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from Visual import plot_svc_decision_function

if __name__ == '__main__':
    # 生成环形数据集（非线性可分）
    # factor 参数控制内圈与外圈半径的比例
    # noise 参数添加随机扰动，使数据更接近真实场景
    X, y = make_circles(n_samples=100, factor=0.1, noise=0.1, random_state=42)

    # 使用 RBF 核的 SVM，C 值较大表示更严格的间隔控制。
    clf = SVC(kernel='rbf', C=1E6)
    clf.fit(X, y)

    # 绘制数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

    # 绘制决策边界（RBF 核在高维空间中找到线性边界，对应原空间的非线性边界）
    plot_svc_decision_function(clf, plot_support=True)

    # 单独强调支持向量（可选），若上一步已经 plot_support=True 则本步可省略
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none', edgecolors='blue')

    plt.title("Kernel SVM (RBF) on Nonlinear Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()