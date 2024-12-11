# -*- coding: utf-8 -*-
# @Time    : 2024/11/25 14:18
# @Author  : EvanWong
# @File    : NB.py
# @Project : MLWork
# @Brief   : 使用朴素贝叶斯算法实现垃圾短信分类。

import math
import pickle
from typing import List, Tuple

from Utils import load_data, create_vocab_list, text_to_vec, train_test_split

# 训练朴素贝叶斯分类器

def train_naive_bayes(train_set: List[Tuple[str, str]], vocab_list: List[str]) -> Tuple[float, float, List[float], List[float]]:
    """
    训练朴素贝叶斯模型，计算先验概率和条件概率。

    :param train_set: 训练集，包含(标签, 文本)的元组列表。
    :param vocab_list: 词汇表列表。
    :return: 先验概率 p_spam 和 p_ham，条件概率列表 p_w_spam 和 p_w_ham。
    """
    num_docs = len(train_set)
    num_words = len(vocab_list)
    # 初始化计数器
    spam_word_count = [0] * num_words
    ham_word_count = [0] * num_words
    spam_doc_count = 0
    # 创建单词到索引的映射
    word_to_index = {word: idx for idx, word in enumerate(vocab_list)}
    for label, text in train_set:
        vec = text_to_vec(vocab_list, text)
        if label == 'spam':
            spam_word_count = [x + y for x, y in zip(spam_word_count, vec)]
            spam_doc_count += 1
        else:
            ham_word_count = [x + y for x, y in zip(ham_word_count, vec)]
    # 计算先验概率
    p_spam = spam_doc_count / num_docs
    p_ham = 1 - p_spam
    # 计算条件概率，使用拉普拉斯平滑
    spam_total_count = sum(spam_word_count) + num_words
    ham_total_count = sum(ham_word_count) + num_words
    p_w_spam = [(count + 1) / spam_total_count for count in spam_word_count]
    p_w_ham = [(count + 1) / ham_total_count for count in ham_word_count]
    return p_spam, p_ham, p_w_spam, p_w_ham

# 保存模型

def save_nb_model(p_spam: float, p_ham: float, p_w_spam: List[float], p_w_ham: List[float], vocab_list: List[str], filename: str = 'naive_bayes_model.pkl'):
    """
    保存训练好的朴素贝叶斯模型到文件。

    :param p_spam: 垃圾短信的先验概率。
    :param p_ham: 正常短信的先验概率。
    :param p_w_spam: 在垃圾短信条件下的单词条件概率列表。
    :param p_w_ham: 在正常短信条件下的单词条件概率列表。
    :param vocab_list: 词汇表列表。
    :param filename: 模型文件名，默认为 'naive_bayes_model.pkl'。
    """
    model = {
        'p_spam': p_spam,
        'p_ham': p_ham,
        'p_w_spam': p_w_spam,
        'p_w_ham': p_w_ham,
        'vocab_list': vocab_list
    }
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved to {filename}')

# 加载模型

def load_nb_model(filename: str = 'naive_bayes_model.pkl') -> Tuple[float, float, List[float], List[float], List[str]]:
    """
    从文件中加载朴素贝叶斯模型。

    :param filename: 模型文件名，默认为 'naive_bayes_model.pkl'。
    :return: 模型的参数，包括先验概率、条件概率和词汇表。
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f'Model loaded from {filename}')
    return model['p_spam'], model['p_ham'], model['p_w_spam'], model['p_w_ham'], model['vocab_list']

# 分类函数

def classify(vec: List[float], p_spam: float, p_ham: float, p_w_spam: List[float], p_w_ham: List[float]) -> str:
    """
    使用朴素贝叶斯模型对输入向量进行分类。

    :param vec: 输入的词向量。
    :param p_spam: 垃圾短信的先验概率。
    :param p_ham: 正常短信的先验概率。
    :param p_w_spam: 在垃圾短信条件下的单词条件概率列表。
    :param p_w_ham: 在正常短信条件下的单词条件概率列表。
    :return: 分类结果，'spam' 或 'ham'。
    """
    # 使用对数避免下溢
    log_p_spam = math.log(p_spam)
    log_p_ham = math.log(p_ham)
    for i in range(len(vec)):
        if vec[i] > 0:
            log_p_spam += vec[i] * math.log(p_w_spam[i])
            log_p_ham += vec[i] * math.log(p_w_ham[i])
    return 'spam' if log_p_spam > log_p_ham else 'ham'

# 评估模型

def evaluate(test_set: List[Tuple[str, str]], vocab_list: List[str], p_spam: float, p_ham: float, p_w_spam: List[float], p_w_ham: List[float]):
    """
    在测试集上评估朴素贝叶斯模型的性能。

    :param test_set: 测试集，包含(标签, 文本)的元组列表。
    :param vocab_list: 词汇表列表。
    :param p_spam: 垃圾短信的先验概率。
    :param p_ham: 正常短信的先验概率。
    :param p_w_spam: 在垃圾短信条件下的单词条件概率列表。
    :param p_w_ham: 在正常短信条件下的单词条件概率列表。
    """
    correct = 0
    total = len(test_set)
    for label, text in test_set:
        vec = text_to_vec(vocab_list, text)
        pred = classify(vec, p_spam, p_ham, p_w_spam, p_w_ham)
        if pred == label:
            correct += 1
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.2f}')

# 预处理新文本

def preprocess_new_text_nb(text: str, vocab_list: List[str]) -> List[float]:
    """
    对新输入的文本进行预处理，转换为模型可用的词向量。

    :param text: 新的文本字符串。
    :param vocab_list: 词汇表列表。
    :return: 预处理后的词向量。
    """
    vec = text_to_vec(vocab_list, text)
    return vec

# 预测新文本

def predict_new_text_nb(text: str):
    """
    使用训练好的朴素贝叶斯模型对新文本进行分类预测。

    :param text: 新的文本字符串。
    """
    p_spam, p_ham, p_w_spam, p_w_ham, vocab_list = load_nb_model()
    vec = preprocess_new_text_nb(text, vocab_list)
    result = classify(vec, p_spam, p_ham, p_w_spam, p_w_ham)
    print(f'The message is predicted to be: {result}')

# 主程序

if __name__ == '__main__':
    # 第一次运行需要训练模型并保存
    m_dataset = load_data('../datasets/SMSSpamCollection')
    m_train_set, m_test_set = train_test_split(m_dataset)
    m_vocab_list = create_vocab_list(m_train_set)
    m_p_spam, m_p_ham, m_p_w_spam, m_p_w_ham = train_naive_bayes(m_train_set, m_vocab_list)
    save_nb_model(m_p_spam, m_p_ham, m_p_w_spam, m_p_w_ham, m_vocab_list)
    evaluate(m_test_set, m_vocab_list, m_p_spam, m_p_ham, m_p_w_spam, m_p_w_ham)

    # 预测新文本
    new_text = input('Please enter a message to classify: ')
    predict_new_text_nb(new_text)
