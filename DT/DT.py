# -*- coding: utf-8 -*-
# @Time    : 2024/10/12 14:10
# @Author  : EvanWong
# @File    : DT.py
# @Project : MLWork


import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import operator
from math import log


def create_dataset():
    """
    创建示例数据集和对应的特征标签。

    :return: data_set - 数据集 (list of lists)
             labels - 特征标签 (list)
    """
    data_set = [
        [0, 0, 0, 0, 'no'],     # 特征数据和分类结果
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
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 特征标签
    return data_set, labels


def create_tree(data_set, labels, feat_labels):
    """
    递归构建决策树。

    :param data_set: 数据集
    :param labels: 特征标签列表
    :param feat_labels: 存储选择的最优特征标签
    :return: 构建的决策树 (dict)
    """
    # 获取当前节点的所有类别标签
    class_list = [example[-1] for example in data_set]
    # 如果类别完全相同，则停止划分，返回该类别
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 如果遍历完所有特征，仍不能将数据集划分成仅包含唯一类别的分组，则返回出现次数最多的类别
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    # 选择最优特征
    best_feature = choose_best_feature_to_split(data_set)
    # 最优特征的标签
    best_feature_label = labels[best_feature]
    feat_labels.append(best_feature_label)
    # 初始化树，使用字典存储
    my_tree = {best_feature_label: {}}
    # 删除已经使用的特征标签
    del(labels[best_feature])
    # 得到列表包含的所有属性值
    feature_values = [example[best_feature] for example in data_set]
    # 去重，得到当前特征所有可能取值
    unique_vals = set(feature_values)
    # 遍历特征的所有取值，递归构建树
    for value in unique_vals:
        sub_labels = labels[:]
        # 递归调用函数，构建子树
        my_tree[best_feature_label][value] = create_tree(
            split_data_set(data_set, best_feature, value),
            sub_labels,
            feat_labels
        )
    return my_tree


def majority_cnt(class_list):
    """
    返回出现次数最多的分类名称。

    :param class_list: 分类名称列表
    :return: 出现次数最多的分类名称
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0  # 初始化类别计数
        class_count[vote] += 1
    # 对类别出现的频率进行排序，降序排列
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1),
                                reverse=True)
    # 返回出现次数最多的类别名称
    return sorted_class_count[0][0]


def choose_best_feature_to_split(data_set):
    """
    选择最好的数据集划分方式。

    :param data_set: 数据集
    :return: 最优特征的索引
    """
    num_features = len(data_set[0]) - 1  # 特征数量（减去标签列）
    base_entropy = calc_shannon_ent(data_set)  # 计算数据集的原始香农熵
    best_info_gain = 0.0  # 最优的信息增益
    best_feature = -1  # 最优特征的索引
    # 遍历所有特征
    for i in range(num_features):
        # 获取第i个特征的所有取值
        feature_list = [example[i] for example in data_set]
        unique_vals = set(feature_list)
        new_entropy = 0.0  # 初始化新的熵
        # 计算每种划分方式的信息熵
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)  # 划分数据集
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        # 计算信息增益
        info_gain = base_entropy - new_entropy
        # 更新信息增益和最优特征
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def split_data_set(data_set, axis, value):
    """
    按照给定特征划分数据集。

    :param data_set: 待划分的数据集
    :param axis: 划分数据集的特征索引
    :param value: 特征的取值
    :return: 划分后的数据集
    """
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            # 去掉axis特征，构建新的特征向量
            reduced_feat_vec = feat_vec[:axis] + feat_vec[axis+1:]
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def calc_shannon_ent(data_set):
    """
    计算给定数据集的香农熵。

    :param data_set: 数据集
    :return: 香农熵
    """
    num_examples = len(data_set)  # 数据集总样本数
    label_counts = {}
    # 统计每个标签出现的次数
    for feat_vec in data_set:
        current_label = feat_vec[-1]  # 提取标签
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    # 计算香农熵
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_examples  # 概率值
        shannon_ent -= prob * log(prob, 2)  # 熵值累加
    return shannon_ent


def get_num_leafs(my_tree):
    """
    获取决策树的叶子节点数目。

    :param my_tree: 决策树
    :return: 叶子节点数目
    """
    num_leafs = 0
    first_str = next(iter(my_tree))  # 获取第一个关键字
    second_dict = my_tree[first_str]  # 获取对应的子树
    for key in second_dict.keys():
        # 判断节点是否是字典，从而判断是否为叶子节点
        if isinstance(second_dict[key], dict):
            num_leafs += get_num_leafs(second_dict[key])  # 递归计算叶子节点
        else:
            num_leafs += 1  # 叶子节点计数加1
    return num_leafs


def get_tree_depth(my_tree):
    """
    获取决策树的深度。

    :param my_tree: 决策树
    :return: 树的深度
    """
    max_depth = 0  # 初始化最大深度
    first_str = next(iter(my_tree))  # 获取第一个关键字
    second_dict = my_tree[first_str]  # 获取对应的子树
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            this_depth = 1 + get_tree_depth(second_dict[key])  # 递归计算深度
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth  # 更新最大深度
    return max_depth


def plot_node(node_txt, center_pt, parent_pt, node_type):
    """
    绘制节点。

    :param node_txt: 节点文本
    :param center_pt: 当前节点坐标
    :param parent_pt: 父节点坐标
    :param node_type: 节点类型（决策节点或叶子节点）
    """
    arrow_args = dict(arrowstyle="<-")  # 定义箭头格式
    font = FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=14)  # 设置字体
    # 绘制节点
    create_plot.ax1.annotate(
        node_txt,
        xy=parent_pt,
        xycoords='axes fraction',
        xytext=center_pt,
        textcoords='axes fraction',
        va="center",
        ha="center",
        bbox=node_type,
        arrowprops=arrow_args,
        fontproperties=font
    )


def plot_mid_text(center_pt, parent_pt, txt_string):
    """
    在父子节点之间填充文本信息。

    :param center_pt: 当前节点坐标
    :param parent_pt: 父节点坐标
    :param txt_string: 标注的文本内容
    """
    x_mid = (parent_pt[0] - center_pt[0]) / 2.0 + center_pt[0]  # 计算文本位置的x坐标
    y_mid = (parent_pt[1] - center_pt[1]) / 2.0 + center_pt[1]  # 计算文本位置的y坐标
    create_plot.ax1.text(x_mid, y_mid, txt_string, va="center", ha="center", rotation=30)


def plot_tree(my_tree, parent_pt, node_txt):
    """
    递归绘制决策树。

    :param my_tree: 决策树
    :param parent_pt: 父节点坐标
    :param node_txt: 节点文本
    """
    decision_node = dict(boxstyle="sawtooth", fc="0.8")  # 决策节点格式
    leaf_node = dict(boxstyle="round4", fc="0.8")  # 叶子节点格式
    num_leafs = get_num_leafs(my_tree)  # 当前子树的叶子节点数目
    depth = get_tree_depth(my_tree)  # 当前子树的深度
    first_str = next(iter(my_tree))  # 根节点标签
    # 计算根节点的位置
    center_pt = (
        plot_tree.x_off + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_w,
        plot_tree.y_off
    )
    # 标注有向边属性值
    plot_mid_text(center_pt, parent_pt, node_txt)
    # 绘制决策节点
    plot_node(first_str, center_pt, parent_pt, decision_node)
    second_dict = my_tree[first_str]  # 子树
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_d  # 下移y坐标
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            # 递归绘制子树
            plot_tree(second_dict[key], center_pt, str(key))
        else:
            # 更新x坐标，绘制叶子节点
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_w
            plot_node(second_dict[key], (plot_tree.x_off, plot_tree.y_off), center_pt, leaf_node)
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), center_pt, str(key))
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_d  # 上移y坐标，回溯到父节点


def create_plot(in_tree):
    """
    创建决策树的绘制面板并启动绘制过程。

    :param in_tree: 决策树
    """
    fig = plt.figure(1, facecolor='white')  # 创建画布
    fig.clf()  # 清空画布
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 无边框，无坐标轴
    # 存储树的宽度和深度
    plot_tree.total_w = float(get_num_leafs(in_tree))
    plot_tree.total_d = float(get_tree_depth(in_tree))
    plot_tree.x_off = -0.5 / plot_tree.total_w  # x轴偏移
    plot_tree.y_off = 1.0  # y轴起始位置
    # 绘制决策树
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()  # 显示绘制结果


if __name__ == '__main__':
    data_set, labels = create_dataset()
    feat_labels = []
    my_tree = create_tree(data_set, labels, feat_labels)
    print(feat_labels)
    create_plot(my_tree)
