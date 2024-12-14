# -*- coding: utf-8 -*-
# @Time    : 2024/11/25 14:18
# @Author  : EvanWong
# @File    : main.py
# @Project : MLWork
# 说明：
# 此文件演示如何整合 BP 和 NB 模型的训练、评估以及最终性能对比。
# 请确保与同一目录下的 BP.py、NB.py、Utils.py 在同一路径内，并有数据集文件。

from Utils import load_data, create_vocab_list, train_test_split
from TermProject.BP import train_nn, save_nn_model, evaluate_nn, predict_new_text_nn
from TermProject.NB import train_naive_bayes, save_nb_model, evaluate, predict_new_text_nb

if __name__ == '__main__':
    # 数据加载
    dataset_path = '../datasets/SMSSpamCollection'
    dataset = load_data(dataset_path)

    if not dataset:
        print("No data loaded. Please check the dataset file and path.")
        exit(1)

    train_set, test_set = train_test_split(dataset, test_ratio=0.2)
    vocab_list = create_vocab_list(train_set)

    # 使用 BP 网络训练
    print("=== Training BP Neural Network ===")
    W1, W2, vocab_list = train_nn(train_set, vocab_list, hidden_size=10, epochs=5)
    save_nn_model(W1, W2, vocab_list)
    bp_accuracy = evaluate_nn(test_set, vocab_list, W1, W2)

    # 使用 Naive Bayes 训练
    print("\n=== Training Naive Bayes ===")
    p_spam, p_ham, p_w_spam, p_w_ham = train_naive_bayes(train_set, vocab_list)
    save_nb_model(p_spam, p_ham, p_w_spam, p_w_ham, vocab_list)
    nb_accuracy = evaluate(test_set, vocab_list, p_spam, p_ham, p_w_spam, p_w_ham)

    # 性能对比
    print("\n=== Performance Comparison ===")
    print(f"BP Neural Network Test Accuracy: {bp_accuracy*100:.2f}%")
    print(f"Naive Bayes Test Accuracy: {nb_accuracy*100:.2f}%")

    if bp_accuracy > nb_accuracy:
        print("BP Neural Network outperforms Naive Bayes on this dataset.")
    elif bp_accuracy < nb_accuracy:
        print("Naive Bayes outperforms BP Neural Network on this dataset.")
    else:
        print("Both models perform equally on this dataset.")

    # 测试新文本分类
    print("\nEnter a message to classify (or 'exit' to quit):")
    while True:
        msg = input("> ")
        if msg.strip().lower() == 'exit':
            break
        print("BP NN Prediction:")
        predict_new_text_nn(msg)
        print("Naive Bayes Prediction:")
        predict_new_text_nb(msg)