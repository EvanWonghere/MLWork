# -*- coding: utf-8 -*-
# @Time    : 2024/11/18 15:06
# @Author  : EvanWong
# @File    : Visual.py
# @Project : MLWork

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def plot_svc_decision_function(model: SVC, ax=None, plot_support: bool = True):
    """
    在给定的坐标轴中绘制 SVC 的决策边界（决策函数）。
    会绘制决策边界和间隔线，并可选地突出显示支持向量。

    参数:
        model: 已训练的 SVC 模型
        ax: 指定要在其上绘制的 matplotlib Axes 对象。如果为 None，则使用当前轴 (plt.gca())。
        plot_support: 是否绘制支持向量位置（默认 True）
    """
    if ax is None:
        ax = plt.gca()  # 若未提供 ax，则获取当前活跃的坐标轴

    # 获取当前坐标轴范围
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    # 在给定范围内创建网格，用于评估决策函数
    x = np.linspace(x_lim[0], x_lim[1], 30)
    y = np.linspace(y_lim[0], y_lim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T

    # 使用模型的决策函数对网格点进行预测
    P = model.decision_function(xy).reshape(X.shape)

    # 绘制决策边界与间隔线（levels=[-1,0,1] 分别为负间隔、决策边界、正间隔）
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1],
               alpha=0.5,
               linestyles=['--', '-', '--'])

    # 绘制支持向量
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1,
                   facecolors='none',
                   edgecolors='black')

    # 恢复坐标轴范围
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)