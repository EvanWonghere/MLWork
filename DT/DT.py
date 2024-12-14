# -*- coding: utf-8 -*-
# @Time    : 2024/10/12 14:10
# @Author  : EvanWong
# @File    : DT.py
# @Project : MLWork

import operator
import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from typing import Any, Dict, List, Tuple, Union


def create_dataset() -> Tuple[List[List[Any]], List[str]]:
    """
    创建示例数据集和对应的特征标签。

    数据集的最后一列为类别标签，其余列为特征值。

    :return:
        dataset: 数据集，每个元素为 [feature_1, feature_2, ..., feature_n, label]
        labels: 特征名称列表，对应 dataset 中除最后一列以外的特征含义
    """
    dataset = [
        [0, 0, 0, 0, 'no'],
        [0, 0, 0, 1, 'no'],
        [0, 1, 0, 1, 'yes'],
        [0, 1, 1, 0, 'yes'],
        [0, 0, 0, 0, 'no'],
        [1, 0, 0, 0, 'no'],
        [1, 0, 0, 1, 'no'],
        [1, 1, 1, 1, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [2, 0, 1, 2, 'yes'],
        [2, 0, 1, 1, 'yes'],
        [2, 1, 0, 1, 'yes'],
        [2, 1, 0, 2, 'yes'],
        [2, 0, 0, 0, 'no']
    ]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataset, labels


def calc_shannon_ent(dataset: List[List[Any]]) -> float:
    """
    计算给定数据集的香农熵 (Shannon Entropy)。

    香农熵用于衡量数据集的不确定度，值越大不确定度越高。
    若数据集为空，熵定义为0。

    :param dataset: 数据集
    :return: 数据集的香农熵
    """
    num_examples = len(dataset)
    if num_examples == 0:
        return 0.0

    label_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        label_counts[current_label] = label_counts.get(current_label, 0) + 1

    shannon_ent = 0.0
    for count in label_counts.values():
        prob = count / num_examples
        shannon_ent -= prob * math.log2(prob)
    return shannon_ent


def split_dataset(dataset: List[List[Any]], feature_index: int, value: Any) -> List[List[Any]]:
    """
    按照给定特征值划分数据集。

    将数据集中指定特征为 feature_index 的值等于 value 的样本选取出来，并去除该特征列。

    :param dataset: 待划分的数据集
    :param feature_index: 待划分的特征索引
    :param value: 划分依据的特征值
    :return: 划分后的子数据集，其中不再包含 feature_index 这一列
    """
    sub_dataset = []
    for feat_vec in dataset:
        if feat_vec[feature_index] == value:
            # 切片去除当前特征列
            reduced_feat_vec = feat_vec[:feature_index] + feat_vec[feature_index + 1:]
            sub_dataset.append(reduced_feat_vec)
    return sub_dataset


def choose_best_feature_to_split(dataset: List[List[Any]]) -> int:
    """
    选择最优特征进行划分，使用信息增益作为评估指标。

    信息增益 = 数据集划分前后的熵差值。
    熵降低最多的特征即为最优特征。

    若无法有效划分，返回 -1。

    :param dataset: 数据集
    :return: 最优特征的索引
    """
    if len(dataset) == 0 or len(dataset[0]) <= 1:
        return -1

    num_features = len(dataset[0]) - 1
    base_entropy = calc_shannon_ent(dataset)
    best_info_gain = 0.0
    best_feature = -1

    for i in range(num_features):
        # 获取当前特征的所有取值并去重
        feature_values = {example[i] for example in dataset}
        new_entropy = 0.0
        for value in feature_values:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / len(dataset)
            new_entropy += prob * calc_shannon_ent(sub_dataset)

        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def majority_cnt(class_list: List[str]) -> str:
    """
    返回列表中出现次数最多的类别。

    当特征用尽仍无法达成完全分类时使用投票表决返回类别。

    :param class_list: 类别名称列表
    :return: 出现次数最多的类别
    """
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0) + 1

    # 对类别按照出现次数降序排序，返回出现次数最多的类别
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(
        dataset: List[List[Any]],
        labels: List[str],
        feat_labels: List[str]
) -> Union[Dict[str, Any], str]:
    """
    递归构建决策树。

    步骤：
    1. 若所有样本类别相同，返回该类别。
    2. 若特征用尽，返回类别出现最多的值。
    3. 否则，选择最优特征划分数据集，创建树的分支并递归构建子树。

    :param dataset: 数据集
    :param labels: 特征标签列表
    :param feat_labels: 存储已选择的特征标签名称，用于后续查询
    :return: 构建的决策树（字典）或类别名称（字符串）
    """
    # 提取数据集的所有类别标签（末列）
    class_list = [example[-1] for example in dataset]

    # 情况1：类别完全相同则直接返回
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 情况2：特征已用尽，无法继续划分，返回最多数投票类别
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)

    # 选择信息增益最大的特征进行划分
    best_feature = choose_best_feature_to_split(dataset)
    if best_feature == -1:
        # 没有特征可用于划分，直接返回多数类别
        return majority_cnt(class_list)

    best_feature_label = labels[best_feature]
    feat_labels.append(best_feature_label)

    # 创建树并从labels中删除已用特征
    my_tree = {best_feature_label: {}}
    del (labels[best_feature])

    # 得到当前最优特征的所有可能取值
    feature_values = {example[best_feature] for example in dataset}

    # 对每个取值递归构建子树
    for value in feature_values:
        sub_labels = labels[:]
        sub_dataset = split_dataset(dataset, best_feature, value)
        my_tree[best_feature_label][value] = create_tree(sub_dataset, sub_labels, feat_labels)

    return my_tree


def get_num_leafs(tree: Dict[str, Any]) -> int:
    """
    获取决策树的叶子节点数目。

    递归遍历整棵树，统计非字典节点的个数。

    :param tree: 决策树
    :return: 叶子节点数目
    """
    num_leafs = 0
    first_str = next(iter(tree))
    second_dict = tree[first_str]
    for key in second_dict:
        if isinstance(second_dict[key], dict):
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(tree: Dict[str, Any]) -> int:
    """
    获取决策树的深度。

    深度为最长路径上节点数的最大值。

    :param tree: 决策树
    :return: 树的深度
    """
    max_depth = 0
    first_str = next(iter(tree))
    second_dict = tree[first_str]
    for key in second_dict:
        if isinstance(second_dict[key], dict):
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        max_depth = max(max_depth, this_depth)
    return max_depth


def plot_node(node_txt: str, center_pt: Tuple[float, float], parent_pt: Tuple[float, float], node_type: Dict[str, Any]):
    """
    绘制节点（决策节点或叶子节点）。

    :param node_txt: 节点文本（特征名称或类别值）
    :param center_pt: 当前节点坐标
    :param parent_pt: 父节点坐标
    :param node_type: 节点图形格式（字典，包含boxstyle与颜色等信息）
    """
    arrow_args = dict(arrowstyle="<-")
    font = FontProperties(fname="/System/Library/Fonts/Supplemental/Songti.ttc", size=14)
    create_plot.ax1.annotate(
        node_txt,
        xy=parent_pt,
        xycoords='axes fraction',
        xytext=center_pt,
        textcoords='axes fraction',
        va="center", ha="center",
        bbox=node_type,
        arrowprops=arrow_args,
        fontproperties=font
    )


def plot_mid_text(center_pt: Tuple[float, float], parent_pt: Tuple[float, float], txt_string: str):
    """
    在父子节点之间填充文本信息（如边的属性值）。

    :param center_pt: 当前节点坐标
    :param parent_pt: 父节点坐标
    :param txt_string: 要绘制的文本（通常是特征取值）
    """
    x_mid = (parent_pt[0] - center_pt[0]) / 2.0 + center_pt[0]
    y_mid = (parent_pt[1] - center_pt[1]) / 2.0 + center_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string, va="center", ha="center", rotation=30)


def plot_tree(tree: Dict[str, Any], parent_pt: Tuple[float, float], node_txt: str):
    """
    递归绘制决策树。

    根据树的结构，不断划分并绘制相应的节点和连线。
    使用全局变量 plot_tree.total_w, plot_tree.total_d, plot_tree.x_off, plot_tree.y_off
    来控制节点分布。

    :param tree: 决策树
    :param parent_pt: 父节点坐标
    :param node_txt: 父子节点连线上显示的文本
    """
    decision_node = dict(boxstyle="sawtooth", fc="0.8")
    leaf_node = dict(boxstyle="round4", fc="0.8")

    num_leafs = get_num_leafs(tree)
    first_str = next(iter(tree))
    center_pt = (
        plot_tree.x_off + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_w,
        plot_tree.y_off
    )

    # 绘制父子节点之间的属性值（边信息）
    plot_mid_text(center_pt, parent_pt, node_txt)
    # 绘制决策节点
    plot_node(first_str, center_pt, parent_pt, decision_node)
    second_dict = tree[first_str]

    # 下移 y 坐标，用于绘制子节点
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_d

    for key in second_dict:
        if isinstance(second_dict[key], dict):
            # 递归绘制子树
            plot_tree(second_dict[key], center_pt, str(key))
        else:
            # 绘制叶子节点
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_w
            plot_node(second_dict[key], (plot_tree.x_off, plot_tree.y_off), center_pt, leaf_node)
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), center_pt, str(key))

    # 子树绘制完成后，上移 y 坐标回到父节点层次
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_d


def create_plot(in_tree: Dict[str, Any]):
    """
    初始化绘图界面并调用函数绘制决策树。

    :param in_tree: 决策树
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()

    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)

    plot_tree.total_w = float(get_num_leafs(in_tree))
    plot_tree.total_d = float(get_tree_depth(in_tree))
    plot_tree.x_off = -0.5 / plot_tree.total_w
    plot_tree.y_off = 1.0

    # 开始绘制决策树
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()


if __name__ == '__main__':
    dataset, labels = create_dataset()
    feat_labels = []
    my_tree = create_tree(dataset, labels[:], feat_labels)  # 使用 labels[:] 复制一份，以免原列表被修改
    print("Feature Labels chosen order:", feat_labels)
    create_plot(my_tree)