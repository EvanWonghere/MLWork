# -*- coding: utf-8 -*-
# @Time    : 2024/11/25 14:35
# @Author  : EvanWong
# @File    : Utils.py
# @Project : MLWork
# @Brief   : 一些实用工具函数，主要用于数据读取与处理。

import re
import random
from typing import List, Tuple

# 数据预处理模块

def load_data(filepath: str) -> List[Tuple[str, str]]:
    """
    从指定路径加载数据集。

    数据集文件中的每一行应为标签和文本内容，以制表符（\t）分隔，例如：
    spam\tCongratulations! You've won a $1000 Walmart gift card. Go to https://bit.ly/12345 to claim now.

    :param filepath: 数据集文件的路径。
    :return: 包含(标签, 文本)元组的列表。
    """
    dataset = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 确保行不为空
                label, text = line.split('\t', 1)
                dataset.append((label, text))
    return dataset

def preprocess_text(text: str) -> List[str]:
    """
    对文本进行预处理，包括去除非字母字符和转换为小写。

    :param text: 需要预处理的文本字符串。
    :return: 预处理后的单词列表。
    """
    # 使用正则表达式替换非字母字符为空格，并将所有字母转换为小写
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = text.split()  # 按空格分割为单词列表
    return words

def create_vocab_list(dataset: List[Tuple[str, str]]) -> List[str]:
    """
    从数据集中创建词汇表。

    词汇表包含数据集中出现的所有唯一单词。

    :param dataset: 包含(标签, 文本)的元组列表的数据集。
    :return: 词汇表列表。
    """
    vocab_set = set()
    for _, text in dataset:
        words = preprocess_text(text)
        vocab_set.update(words)  # 将单词添加到集合中，自动去重
    return list(vocab_set)

def text_to_vec(vocab_list: List[str], text: str) -> List[float]:
    """
    将文本转换为词向量。

    词向量的长度与词汇表相同，每个元素表示对应单词在文本中出现的次数。

    :param vocab_list: 词汇表列表。
    :param text: 需要转换的文本字符串。
    :return: 表示文本的词频向量。
    """
    words = preprocess_text(text)
    vec = [0.0] * len(vocab_list)
    # 创建单词到索引的映射，提升查找效率
    word_to_index = {word: idx for idx, word in enumerate(vocab_list)}
    for word in words:
        if word in word_to_index:
            idx = word_to_index[word]
            vec[idx] += 1.0
    return vec

def train_test_split(dataset: List[Tuple[str, str]], test_ratio: float = 0.2) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    将数据集划分为训练集和测试集。

    :param dataset: 需要划分的数据集。
    :param test_ratio: 测试集所占的比例，默认为0.2（20%）。
    :return: 训练集和测试集的元组。
    """
    dataset_copy = dataset.copy()  # 复制数据集，避免修改原数据
    random.shuffle(dataset_copy)  # 随机打乱数据集
    test_size = int(len(dataset_copy) * test_ratio)
    test_set = dataset_copy[:test_size]
    train_set = dataset_copy[test_size:]
    return train_set, test_set
