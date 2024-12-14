# -*- coding: utf-8 -*-
# @Time    : 2024/11/25 14:18
# @Author  : EvanWong
# @File    : NB.py
# @Project : MLWork
# @Brief   : 使用朴素贝叶斯算法实现垃圾短信分类。

import math
import pickle
from typing import List, Tuple

from tqdm import tqdm

from Utils import load_data, create_vocab_list, text_to_vec, train_test_split

def train_naive_bayes(train_set: List[Tuple[str, str]], vocab_list: List[str]) -> Tuple[float, float, List[float], List[float]]:
    """
    训练朴素贝叶斯模型并显示训练进度。
    """
    num_docs = len(train_set)
    num_words = len(vocab_list)
    spam_word_count = [0] * num_words
    ham_word_count = [0] * num_words
    spam_doc_count = 0

    # 使用 tqdm 显示处理进度
    for label, text in tqdm(train_set, desc="Training Naive Bayes", ncols=80):
        vec = text_to_vec(vocab_list, text)
        if label == 'spam':
            spam_word_count = [x + y for x, y in zip(spam_word_count, vec)]
            spam_doc_count += 1
        else:
            ham_word_count = [x + y for x, y in zip(ham_word_count, vec)]

    p_spam = spam_doc_count / num_docs if num_docs > 0 else 0.5
    p_ham = 1 - p_spam

    spam_total_count = sum(spam_word_count) + num_words
    ham_total_count = sum(ham_word_count) + num_words
    p_w_spam = [(count + 1) / spam_total_count for count in spam_word_count]
    p_w_ham = [(count + 1) / ham_total_count for count in ham_word_count]

    print("[NB] Training Completed. Learned Prior and Conditional probabilities.")
    return p_spam, p_ham, p_w_spam, p_w_ham

def save_nb_model(p_spam: float, p_ham: float, p_w_spam: List[float], p_w_ham: List[float], vocab_list: List[str], filename: str = 'naive_bayes_model.pkl'):
    model = {
        'p_spam': p_spam,
        'p_ham': p_ham,
        'p_w_spam': p_w_spam,
        'p_w_ham': p_w_ham,
        'vocab_list': vocab_list
    }
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f'[NB] Model saved to {filename}')

def load_nb_model(filename: str = 'naive_bayes_model.pkl') -> Tuple[float, float, List[float], List[float], List[str]]:
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f'[NB] Model loaded from {filename}')
    return model['p_spam'], model['p_ham'], model['p_w_spam'], model['p_w_ham'], model['vocab_list']

def classify(vec: List[float], p_spam: float, p_ham: float, p_w_spam: List[float], p_w_ham: List[float]) -> str:
    log_p_spam = math.log(p_spam + 1e-9)  # 加小值防止log(0)
    log_p_ham = math.log(p_ham + 1e-9)
    for i in range(len(vec)):
        if vec[i] > 0:
            log_p_spam += vec[i] * math.log(p_w_spam[i])
            log_p_ham += vec[i] * math.log(p_w_ham[i])
    return 'spam' if log_p_spam > log_p_ham else 'ham'

def evaluate(test_set: List[Tuple[str, str]], vocab_list: List[str], p_spam: float, p_ham: float, p_w_spam: List[float], p_w_ham: List[float]):
    correct = 0
    total = len(test_set)
    for label, text in test_set:
        vec = text_to_vec(vocab_list, text)
        pred = classify(vec, p_spam, p_ham, p_w_spam, p_w_ham)
        if pred == label:
            correct += 1
    accuracy = correct / total if total > 0 else 0
    print(f'[NB] Test Accuracy: {accuracy*100:.2f}%')
    return accuracy

def preprocess_new_text_nb(text: str, vocab_list: List[str]) -> List[float]:
    vec = text_to_vec(vocab_list, text)
    return vec

def predict_new_text_nb(text: str):
    p_spam, p_ham, p_w_spam, p_w_ham, vocab_list = load_nb_model()
    vec = preprocess_new_text_nb(text, vocab_list)
    result = classify(vec, p_spam, p_ham, p_w_spam, p_w_ham)
    print(f'[NB] The message is predicted to be: {result}')