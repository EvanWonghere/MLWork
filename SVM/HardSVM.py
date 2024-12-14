# -*- coding: utf-8 -*-
# @Time    : 2024/11/18 14:10
# @Author  : EvanWong
# @File    : HardSVM.py
# @Project : MLWork

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

from Visual import plot_svc_decision_function

if __name__ == '__main__':
    """
    实验说明:
    使用 Hard Margin SVM 在可线性分隔的数据集上训练模型，并可视化决策边界和支持向量。
    """

    # 生成可分离的两类数据点
    X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

    # 使用线性核和非常大的 C 值训练 Hard Margin SVM（C 值越大，对误分类的惩罚越严）
    model = SVC(kernel='linear', C=1E10)
    model.fit(X, y)

    # 绘制数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')

    # 绘制决策函数及支持向量
    plot_svc_decision_function(model, plot_support=True)

    # 显示结果图
    plt.title("Hard Margin SVM Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()