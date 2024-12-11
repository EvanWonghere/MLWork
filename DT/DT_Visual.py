# -*- coding: utf-8 -*-
# @Time    : 2024/10/12 15:28
# @Author  : EvanWong
# @File    : DT_Visual.py
# @Project : MLWork


def get_num_leafs(myTree):
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += get_num_leafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs
