# -*- coding: utf-8 -*-
# @Time    : 2024/10/28 14:58
# @Author  : EvanWong
# @File    : BP.py
# @Project : MLWork

import numpy as np
import tensorflow as tf


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid 激活函数。

    :param x: 输入数组
    :return: 经 Sigmoid 激活后的输出数组
    """
    return 1 / (1 + np.exp(-x))


def calc_error(e: np.ndarray) -> float:
    """
    计算误差 (0.5 * sum(e^2))。

    :param e: 误差向量 (预测值 - 真实值)
    :return: 误差标量值
    """
    return 0.5 * np.dot(e, e)


# ==================== 数据加载部分（使用 TensorFlow） ====================
print("加载 MNIST 数据集（使用 TensorFlow）...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 数据预处理：将图像展开为一维向量，并归一化到 [0,1]
x_train = x_train.reshape(x_train.shape[0], 28*28).astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32') / 255.0

print(f"训练集大小: {x_train.shape}, 测试集大小: {x_test.shape}")

# ============= 神经网络配置 =============
samp_num = x_train.shape[0]       # 样本总数
inp_num = x_train.shape[1]        # 输入层节点数（特征数）
out_num = 10                       # 输出节点数（0~9数字分类）
hid_num = 9                        # 隐层节点数(可调)
w1 = 0.2 * np.random.random((inp_num, hid_num)) - 0.1  # 输入层->隐层权重
w2 = 0.2 * np.random.random((hid_num, out_num)) - 0.1  # 隐层->输出层权重
hid_offset = np.zeros(hid_num)     # 隐层偏置
out_offset = np.zeros(out_num)     # 输出层偏置
inp_lrate = 0.3                    # 输入层权值学习率
hid_lrate = 0.3                    # 隐层权值学习率
err_th = 0.01                      # 学习误差门限（本代码未使用，可自行实现早停条件）

# ============= 训练过程（单次遍历全部训练数据） =============
print("开始训练...")

for idx in range(samp_num):
    # 构造 one-hot 向量作为真实标签
    t_label = np.zeros(out_num)
    t_label[y_train[idx]] = 1

    # 前向传播
    hid_value = np.dot(x_train[idx], w1) + hid_offset   # 隐层输入
    hid_act = sigmoid(hid_value)                        # 隐层激活输出
    out_value = np.dot(hid_act, w2) + out_offset        # 输出层输入
    out_act = sigmoid(out_value)                        # 输出层激活输出

    # 反向传播
    e = t_label - out_act
    out_delta = e * out_act * (1 - out_act)
    hid_delta = hid_act * (1 - hid_act) * np.dot(w2, out_delta)

    # 更新权重和偏置
    w2 += hid_lrate * np.outer(hid_act, out_delta)
    w1 += inp_lrate * np.outer(x_train[idx], hid_delta)
    out_offset += hid_lrate * out_delta
    hid_offset += inp_lrate * hid_delta

    # 若需要，可基于误差实现早停（可选）
    # current_error = calc_error(e)
    # if current_error < err_th:
    #     print(f"在 {idx} 次训练后误差低于阈值，提前结束训练.")
    #     break

print("训练结束")

# ============= 测试网络 =============
print("开始测试...")
numbers = np.zeros(out_num, dtype=int)
right = np.zeros(out_num, dtype=int)

# 统计测试集中各数字的数目
for lbl in y_test:
    numbers[lbl] += 1

for i in range(len(x_test)):
    hid_value = np.dot(x_test[i], w1) + hid_offset
    hid_act = sigmoid(hid_value)
    out_value = np.dot(hid_act, w2) + out_offset
    out_act = sigmoid(out_value)
    pred = np.argmax(out_act)
    if pred == y_test[i]:
        right[y_test[i]] += 1

print("各类别预测正确数:", right)
print("各类别样本数:", numbers)

result = right / numbers
accuracy = right.sum() / len(x_test)
print("各类别准确率:", result)
print("总体准确率:", accuracy)

# ============= 保存网络参数 =============
print("保存网络参数到 MyNetWork 文件...")
with open("MyNetWork", 'w') as net_file:
    net_file.write(f"{inp_num}\n")
    net_file.write(f"{hid_num}\n")
    net_file.write(f"{out_num}\n")
    for row in w1:
        net_file.write(" ".join(map(str, row)) + "\n")
    net_file.write("\n")
    for row in w2:
        net_file.write(" ".join(map(str, row)) + "\n")

print("保存完毕。")