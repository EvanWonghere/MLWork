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
from tqdm import tqdm

from Utils import load_data, create_vocab_list, text_to_vec, train_test_split

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def d_sigmoid(y: float) -> float:
    return y * (1 - y)

def init_weights(input_size: int, hidden_size: int, output_size: int) -> Tuple[List[List[float]], List[List[float]]]:
    W1 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
    W2 = [[random.uniform(-0.5, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]
    return W1, W2

def forward(x: List[float], W1: List[List[float]], W2: List[List[float]]) -> Tuple[List[float], List[float]]:
    hidden_input = [sum(x[i] * W1[i][j] for i in range(len(x))) for j in range(len(W1[0]))]
    hidden_output = [sigmoid(h) for h in hidden_input]

    final_input = [sum(hidden_output[j] * W2[j][k] for j in range(len(hidden_output))) for k in range(len(W2[0]))]
    final_output = [sigmoid(f) for f in final_input]
    return hidden_output, final_output

def backward(x: List[float], y: List[int],
             hidden_output: List[float], final_output: List[float],
             W1: List[List[float]], W2: List[List[float]],
             learning_rate: float = 0.1):
    output_deltas = [(y[k] - final_output[k]) * d_sigmoid(final_output[k]) for k in range(len(y))]

    for j in range(len(W2)):
        for k in range(len(W2[0])):
            W2[j][k] += learning_rate * output_deltas[k] * hidden_output[j]

    hidden_deltas = [d_sigmoid(hidden_output[j]) * sum(output_deltas[k] * W2[j][k] for k in range(len(output_deltas)))
                     for j in range(len(hidden_output))]

    for i in range(len(W1)):
        for j in range(len(W1[0])):
            W1[i][j] += learning_rate * hidden_deltas[j] * x[i]

def train_nn(train_set: List[Tuple[str, str]], vocab_list: List[str], hidden_size: int = 10, epochs: int = 5) \
        -> Tuple[List[List[float]], List[List[float]], List[str]]:
    input_size = len(vocab_list)
    output_size = 1  # 二分类
    W1, W2 = init_weights(input_size, hidden_size, output_size)

    # 每轮训练后评估训练集准确率或误差
    for epoch in range(epochs):
        random.shuffle(train_set)
        # 使用tqdm显示进度
        total_loss = 0.0
        correct = 0
        for label, text in tqdm(train_set, desc=f"Epoch {epoch+1}/{epochs}", ncols=80):
            x = text_to_vec(vocab_list, text)
            norm_factor = max(1, sum(x))
            x = [xi / norm_factor for xi in x]
            y = [1] if label == 'spam' else [0]
            hidden_output, final_output = forward(x, W1, W2)
            loss = (y[0] - final_output[0])**2 / 2
            total_loss += loss
            pred = 'spam' if final_output[0] > 0.5 else 'ham'
            if pred == label:
                correct += 1
            backward(x, y, hidden_output, final_output, W1, W2)
        avg_loss = total_loss / len(train_set)
        accuracy = correct / len(train_set)
        print(f"Epoch {epoch+1}/{epochs}: Avg Loss={avg_loss:.4f}, Training Accuracy={accuracy*100:.2f}%")

    return W1, W2, vocab_list

def save_nn_model(W1: List[List[float]], W2: List[List[float]], vocab_list: List[str], filename: str = 'bp_nn_model.pkl'):
    model = {
        'W1': W1,
        'W2': W2,
        'vocab_list': vocab_list
    }
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved to {filename}')

def load_nn_model(filename: str = 'bp_nn_model.pkl') -> Tuple[List[List[float]], List[List[float]], List[str]]:
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f'Model loaded from {filename}')
    return model['W1'], model['W2'], model['vocab_list']

def predict(x: List[float], W1: List[List[float]], W2: List[List[float]]) -> str:
    _, final_output = forward(x, W1, W2)
    return 'spam' if final_output[0] > 0.5 else 'ham'

def evaluate_nn(test_set: List[Tuple[str, str]], vocab_list: List[str], W1: List[List[float]], W2: List[List[float]]):
    correct = 0
    total = len(test_set)
    for label, text in test_set:
        x = text_to_vec(vocab_list, text)
        x = [xi / max(1, sum(x)) for xi in x]
        pred = predict(x, W1, W2)
        if pred == label:
            correct += 1
    accuracy = correct / total
    print(f'[BP NN] Test Accuracy: {accuracy*100:.2f}%')
    return accuracy

def preprocess_new_text_nn(text: str, vocab_list: List[str]) -> List[float]:
    vec = text_to_vec(vocab_list, text)
    x = [xi / max(1, sum(vec)) for xi in vec]
    return x

def predict_new_text_nn(text: str):
    W1, W2, vocab_list = load_nn_model()
    x = preprocess_new_text_nn(text, vocab_list)
    result = predict(x, W1, W2)
    print(f'[BP NN] The message is predicted to be: {result}')