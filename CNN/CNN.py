# -*- coding: utf-8 -*-
# @Time    : 2024/11/4 14:48
# @Author  : EvanWong
# @File    : CNN.py
# @Project : MLWork

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from typing import Optional, Dict


class StopAtHighAccuracy(tf.keras.callbacks.Callback):
    """
    当模型在训练集上达到 99.5% 以上准确率时，停止训练的回调函数。
    """
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        logs = logs or {}
        accuracy = logs.get('accuracy')
        if accuracy and accuracy > 0.995:
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True


def plot_training_history(history: tf.keras.callbacks.History):
    """
    绘制训练过程中 Loss 和 Accuracy 的曲线（包含训练集和验证集）。

    :param history: 模型训练历史对象
    """
    fig, ax = plt.subplots(2, 1, figsize=(8, 10))

    # Plot Loss
    ax[0].plot(history.history['loss'], color='b', label="Training Loss")
    ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss")
    ax[0].legend(loc='best', shadow=True)
    ax[0].set_title("Loss Curve")
    ax[0].set_ylabel("Loss")
    ax[0].set_xlabel("Epoch")

    # Plot Accuracy
    ax[1].plot(history.history['accuracy'], color='b', label="Training Accuracy")
    ax[1].plot(history.history['val_accuracy'], color='r', label="Validation Accuracy")
    ax[1].legend(loc='best', shadow=True)
    ax[1].set_title("Accuracy Curve")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlabel("Epoch")

    plt.tight_layout()
    plt.show()


def main():
    # 加载 MNIST 数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 数据预处理：归一化并增加通道维度
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255.0

    # 标签 One-Hot 编码
    y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
    y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

    # 模型参数设定
    input_shape = (28, 28, 1)
    num_classes = 10
    batch_size = 64
    epochs = 5

    # 构建 CNN 模型
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(strides=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 设置回调函数，当准确率达到99.5%时停止训练
    callbacks = [StopAtHighAccuracy()]

    # 开始训练
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks
    )

    # 绘制训练曲线
    plot_training_history(history)

    # 测试集评估
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # 对测试集进行预测
    y_pred_proba = model.predict(x_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 计算混淆矩阵
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt='g', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


if __name__ == '__main__':
    main()