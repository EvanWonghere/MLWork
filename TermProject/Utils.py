# -*- coding: utf-8 -*-
# @Time    : 2024/11/25 14:35
# @Author  : EvanWong
# @File    : Utils.py
# @Project : MLWork
# @Brief   : 一些实用工具函数，主要用于数据读取与处理。

import re
import random
from typing import List, Tuple

def load_data(filepath: str) -> List[Tuple[str, str]]:
    """
    从指定路径加载数据集。
    数据格式：
    label<TAB>text
    label 可为 'spam' 或 'ham'。
    """
    dataset = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    label, text = line.split('\t', 1)
                    dataset.append((label, text))
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Please check the path.")
    return dataset

def preprocess_text(text: str) -> List[str]:
    """对文本进行预处理：移除非字母字符并转换为小写，然后分词。"""
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    return words

def create_vocab_list(dataset: List[Tuple[str, str]]) -> List[str]:
    """
    从数据集中创建词汇表。
    词汇表包含数据集中出现的所有唯一单词。
    """
    vocab_set = set()
    for _, text in dataset:
        words = preprocess_text(text)
        vocab_set.update(words)
    return list(vocab_set)

def text_to_vec(vocab_list: List[str], text: str) -> List[float]:
    """
    将文本转换为词向量。向量长度与词汇表相同，元素为该单词在文本中出现的次数。
    """
    words = preprocess_text(text)
    vec = [0.0] * len(vocab_list)
    word_to_index = {word: idx for idx, word in enumerate(vocab_list)}
    for word in words:
        if word in word_to_index:
            vec[word_to_index[word]] += 1.0
    return vec

def train_test_split(dataset: List[Tuple[str, str]], test_ratio: float = 0.2) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """将数据集划分为训练集和测试集。"""
    if not dataset:
        print("Warning: The dataset is empty.")
        return [], []
    dataset_copy = dataset.copy()
    random.shuffle(dataset_copy)
    test_size = int(len(dataset_copy) * test_ratio)
    test_set = dataset_copy[:test_size]
    train_set = dataset_copy[test_size:]
    return train_set, test_set