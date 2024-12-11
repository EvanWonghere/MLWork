# -*- coding: utf-8 -*-
# @Time    : 2024/9/23 14:36
# @Author  : EvanWong
# @File    : LDA.py
# @Project : MLWork

import numpy as np
import matplotlib.pyplot as plt

# 创建数据集，包含特征值和对应的类别标签
dataset = np.array([
    [0.666, 0.091, 1],  # 样本1，特征值为(0.666, 0.091)，类别标签为1
    [0.243, 0.267, 1],  # 样本2
    [0.244, 0.056, 1],  # 样本3
    [0.342, 0.098, 1],  # 样本4
    [0.638, 0.16, 1],   # 样本5
    [0.656, 0.197, 1],  # 样本6
    [0.359, 0.369, 1],  # 样本7
    [0.592, 0.041, 1],  # 样本8
    [0.718, 0.102, 1],  # 样本9
    [0.697, 0.46, 0],   # 样本10，类别标签为0
    [0.774, 0.376, 0],  # 样本11
    [0.633, 0.263, 0],  # 样本12
    [0.607, 0.317, 0],  # 样本13
    [0.555, 0.214, 0],  # 样本14
    [0.402, 0.236, 0],  # 样本15
    [0.481, 0.149, 0],  # 样本16
    [0.436, 0.21, 0],   # 样本17
    [0.557, 0.216, 0]   # 样本18
])

# 提取特征矩阵X和标签向量Y
X = dataset[:, :2]  # 取所有样本的前两列作为特征矩阵X
Y = dataset[:, 2]   # 取第三列作为标签向量Y

# 计算类别0和类别1的均值向量mu_0和mu_1
mu_0 = np.mean(X[Y == 0], axis=0)  # 对类别为0的样本求均值
mu_1 = np.mean(X[Y == 1], axis=0)  # 对类别为1的样本求均值

# 初始化类内散度矩阵S_w为2x2的零矩阵
S_w = np.zeros((2, 2))

# 计算类内散度矩阵S_w
for i in range(len(X)):
    if Y[i] == 0:
        # 对于类别0的样本，计算(X_i - mu_0)*(X_i - mu_0)^T并累加到S_w
        deviation = X[i] - mu_0  # 计算偏差向量
        S_w += np.outer(deviation, deviation)  # 外积并累加
    else:
        # 对于类别1的样本，计算(X_i - mu_1)*(X_i - mu_1)^T并累加到S_w
        deviation = X[i] - mu_1
        S_w += np.outer(deviation, deviation)

# 计算线性判别分析（LDA）的权重向量w_prime
# w_prime = S_w的逆矩阵 * (mu_0 - mu_1)
w_prime = np.linalg.inv(S_w).dot(mu_0 - mu_1)

# 计算两个类别均值向量的中点，用于决策边界的绘制
mid_point = (mu_0 + mu_1) / 2

# 计算投影方向w_prime的斜率
slope = w_prime[1] / w_prime[0]

# 计算决策边界的斜率和截距（决策边界垂直于投影方向）
slope_perpendicular = -1 / slope  # 垂直线的斜率为原斜率的负倒数
intercept_perpendicular = mid_point[1] - slope_perpendicular * mid_point[0]  # 截距计算

# 创建绘图窗口，设置大小为10x10英寸
plt.figure(figsize=(10, 10))

# 绘制类别1（正类）样本的散点图，蓝色表示
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], c="b", label="Positive")
# 绘制类别0（负类）样本的散点图，红色表示
plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], c="r", label="Negative")

# 绘制LDA的决策边界
x_values = np.linspace(0, 1, 100)  # 定义x轴的取值范围，从0到1，取100个点
y_values = slope_perpendicular * x_values + intercept_perpendicular  # 根据斜率和截距计算y值
plt.plot(x_values, y_values, 'g', label="LDA")  # 绘制决策边界，绿色线条

# 设置图形的x轴和y轴标签
plt.xlabel("Attr 1")  # 特征1
plt.ylabel("Attr 2")  # 特征2

# 显示图例，标识各类别和决策边界
plt.legend()

# 设置图形标题
plt.title("Dataset and LDA")

# 显示绘制的图形
plt.show()

# 输出计算得到的投影向量w_prime
print(f'Projected vector is：{w_prime}')
# 输出LDA决策边界的函数表达式
print(f'Function of LDA is：y = {slope_perpendicular}x + {intercept_perpendicular}')
