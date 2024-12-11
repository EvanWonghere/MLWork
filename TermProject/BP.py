# -*- coding: utf-8 -*-
# @Time    : 2024/11/25 14:18
# @Author  : EvanWong
# @File    : BP.py
# @Project : MLWork
# @Brief   : 使用 BP 神经网络实现垃圾短信分类。

import math
import random
import pickle
from typing import List, Tuple

from Utils import load_data, create_vocab_list, text_to_vec, train_test_split

# 定义激活函数和其导数

def sigmoid(x: float) -> float:
    """
    Sigmoid 激活函数。

    定义为：
    f(x) = 1 / (1 + exp(-x))

    :param x: 输入值。
    :return: 激活函数的输出值。
    """
    return 1 / (1 + math.exp(-x))

def d_sigmoid(y: float) -> float:
    """
    Sigmoid 激活函数的导数。

    定义为：
    f'(x) = f(x) * (1 - f(x))

    注意，这里的输入 y 已经是 f(x) 的输出。

    :param y: Sigmoid 函数的输出值。
    :return: 激活函数导数的值。
    """
    return y * (1 - y)

# 初始化权重

def init_weights(input_size: int, hidden_size: int, output_size: int) -> Tuple[List[List[float]], List[List[float]]]:
    """
    初始化神经网络的权重矩阵。

    使用均匀分布随机初始化权重，范围在 -0.5 到 0.5 之间。

    :param input_size: 输入层节点数。
    :param hidden_size: 隐藏层节点数。
    :param output_size: 输出层节点数。
    :return: 输入层到隐藏层的权重矩阵 W1，隐藏层到输出层的权重矩阵 W2。
    """
    W1 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
    W2 = [[random.uniform(-0.5, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]
    return W1, W2

# 前向传播

def forward(x: List[float], W1: List[List[float]], W2: List[List[float]]) -> Tuple[List[float], List[float]]:
    """
    执行前向传播计算。

    :param x: 输入向量。
    :param W1: 输入层到隐藏层的权重矩阵。
    :param W2: 隐藏层到输出层的权重矩阵。
    :return: 隐藏层的输出值列表，输出层的输出值列表。
    """
    # 输入层到隐藏层
    hidden_input = [sum(x[i] * W1[i][j] for i in range(len(x))) for j in range(len(W1[0]))]
    hidden_output = [sigmoid(h) for h in hidden_input]
    # 隐藏层到输出层
    final_input = [sum(hidden_output[j] * W2[j][k] for j in range(len(hidden_output))) for k in range(len(W2[0]))]
    final_output = [sigmoid(f) for f in final_input]
    return hidden_output, final_output

# 反向传播

def backward(x: List[float], y: List[int],
             hidden_output: List[float], final_output: List[float],
             W1: List[List[float]], W2: List[List[float]],
             learning_rate: float = 0.1):
    """
    执行反向传播算法，更新权重矩阵。

    :param x: 输入向量。
    :param y: 目标输出值列表。
    :param hidden_output: 隐藏层的输出值列表。
    :param final_output: 输出层的输出值列表。
    :param W1: 输入层到隐藏层的权重矩阵。
    :param W2: 隐藏层到输出层的权重矩阵。
    :param learning_rate: 学习率，控制权重更新的步长。
    """
    # 输出层误差
    output_deltas = [(y[k] - final_output[k]) * d_sigmoid(final_output[k]) for k in range(len(y))]
    # 更新隐藏层到输出层的权重
    for j in range(len(W2)):
        for k in range(len(W2[0])):
            W2[j][k] += learning_rate * output_deltas[k] * hidden_output[j]
    # 隐藏层误差
    hidden_deltas = [d_sigmoid(hidden_output[j]) * sum(output_deltas[k] * W2[j][k] for k in range(len(output_deltas)))
                     for j in range(len(hidden_output))]
    # 更新输入层到隐藏层的权重
    for i in range(len(W1)):
        for j in range(len(W1[0])):
            W1[i][j] += learning_rate * hidden_deltas[j] * x[i]

# 训练神经网络

def train_nn(train_set: List[Tuple[str, str]], vocab_list: List[str], hidden_size: int = 10, epochs: int = 5) \
        -> Tuple[List[List[float]], List[List[float]], List[str]]:
    """
    训练 BP 神经网络模型。

    :param train_set: 训练集，包含(标签, 文本)的元组列表。
    :param vocab_list: 词汇表列表。
    :param hidden_size: 隐藏层节点数，默认为10。
    :param epochs: 训练轮数，默认为5。
    :return: 训练好的权重矩阵 W1 和 W2，以及词汇表。
    """
    input_size = len(vocab_list)
    output_size = 1  # 二分类，输出节点为1
    W1, W2 = init_weights(input_size, hidden_size, output_size)
    for epoch in range(epochs):
        random.shuffle(train_set)  # 每个 epoch 打乱训练集
        for label, text in train_set:
            x = text_to_vec(vocab_list, text)
            # 输入向量归一化处理，避免数值过大
            x = [xi / max(1, sum(x)) for xi in x]
            y = [1] if label == 'spam' else [0]
            hidden_output, final_output = forward(x, W1, W2)
            backward(x, y, hidden_output, final_output, W1, W2)
        print(f'Epoch {epoch + 1} completed')
    return W1, W2, vocab_list

# 保存模型

def save_nn_model(W1: List[List[float]], W2: List[List[float]], vocab_list: List[str], filename: str = 'bp_nn_model.pkl'):
    """
    保存训练好的神经网络模型到文件。

    :param W1: 输入层到隐藏层的权重矩阵。
    :param W2: 隐藏层到输出层的权重矩阵。
    :param vocab_list: 词汇表列表。
    :param filename: 模型文件名，默认为 'bp_nn_model.pkl'。
    """
    model = {
        'W1': W1,
        'W2': W2,
        'vocab_list': vocab_list
    }
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved to {filename}')

# 加载模型

def load_nn_model(filename: str = 'bp_nn_model.pkl') -> Tuple[List[List[float]], List[List[float]], List[str]]:
    """
    从文件中加载神经网络模型。

    :param filename: 模型文件名，默认为 'bp_nn_model.pkl'。
    :return: 加载的权重矩阵 W1 和 W2，以及词汇表。
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f'Model loaded from {filename}')
    return model['W1'], model['W2'], model['vocab_list']

# 预测函数

def predict(x: List[float], W1: List[List[float]], W2: List[List[float]]) -> str:
    """
    使用训练好的模型对输入向量进行预测。

    :param x: 输入向量。
    :param W1: 输入层到隐藏层的权重矩阵。
    :param W2: 隐藏层到输出层的权重矩阵。
    :return: 预测结果，'spam' 或 'ham'。
    """
    _, final_output = forward(x, W1, W2)
    return 'spam' if final_output[0] > 0.5 else 'ham'

# 评估模型

def evaluate_nn(test_set: List[Tuple[str, str]], vocab_list: List[str], W1: List[List[float]], W2: List[List[float]]):
    """
    在测试集上评估模型的性能。

    :param test_set: 测试集，包含(标签, 文本)的元组列表。
    :param vocab_list: 词汇表列表。
    :param W1: 输入层到隐藏层的权重矩阵。
    :param W2: 隐藏层到输出层的权重矩阵。
    """
    correct = 0
    total = len(test_set)
    for label, text in test_set:
        x = text_to_vec(vocab_list, text)
        x = [xi / max(1, sum(x)) for xi in x]
        pred = predict(x, W1, W2)
        if pred == label:
            correct += 1
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.2f}')

# 预处理新文本

def preprocess_new_text_nn(text: str, vocab_list: List[str]) -> List[float]:
    """
    对新输入的文本进行预处理，转换为模型可用的输入向量。

    :param text: 新的文本字符串。
    :param vocab_list: 词汇表列表。
    :return: 预处理后的输入向量。
    """
    vec = text_to_vec(vocab_list, text)
    x = [xi / max(1, sum(vec)) for xi in vec]
    return x

# 预测新文本

def predict_new_text_nn(text: str):
    """
    使用训练好的模型对新文本进行分类预测。

    :param text: 新的文本字符串。
    """
    W1, W2, vocab_list = load_nn_model()
    x = preprocess_new_text_nn(text, vocab_list)
    result = predict(x, W1, W2)
    print(f'The message is predicted to be: {result}')

# 主程序

if __name__ == '__main__':
    # 第一次运行需要训练模型并保存
    m_dataset = load_data('../datasets/SMSSpamCollection')
    m_train_set, m_test_set = train_test_split(m_dataset)
    m_vocab_list = create_vocab_list(m_train_set)
    mW1, mW2, m_vocab_list = train_nn(m_train_set, m_vocab_list)
    save_nn_model(mW1, mW2, m_vocab_list)
    evaluate_nn(m_test_set, m_vocab_list, mW1, mW2)

    # 预测新文本
    new_text = input('Please enter a message to classify: ')
    predict_new_text_nn(new_text)
