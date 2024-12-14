# -*- coding: utf-8 -*-
# @Time    : 2024/11/18 14:10
# @Author  : EvanWong
# @File    : SoftSVM.py
# @Project : MLWork
#
# 文件说明：
# 本文件对比了不同 C 值（正则化强度）对 Soft Margin SVM 决策边界的影响。
# 当数据不能完美线性分离时，Soft Margin SVM 允许一定程度的误分类，
# C 值越小，模型对误分类点越容忍，间隔更宽松；C 值越大，则越接近 Hard Margin。

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

from Visual import plot_svc_decision_function

if __name__ == '__main__':
    # 生成二分类数据集，但类分布可能有一定重叠（不完全可分）
    X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.8)

    # 创建两个子图用于比较不同 C 值下的决策函数
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(left=0.06, right=0.95, wspace=0.1)

    # 两个不同的 C 值
    C_values = [10, 0.1]

    for i, C_val in enumerate(C_values):
        # 使用线性核进行分类
        model = SVC(kernel="linear", C=C_val)
        model.fit(X, y)

        # 在对应子图上绘制样本点及决策边界
        ax = axes[i]
        ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="summer")
        plot_svc_decision_function(model, ax=ax, plot_support=True)

        # 设置子图标题展示 C 值
        ax.set_title(f'C = {C_val:.1f}', size=14)
        ax.set_xlabel("Feature 1")
        if i == 0:
            ax.set_ylabel("Feature 2")

    plt.suptitle("Soft Margin SVM with Different C Values", size=16)
    plt.show()