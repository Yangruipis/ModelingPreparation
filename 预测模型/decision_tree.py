# -*- coding:utf8 -*-
import pandas as pd
from collections import Counter


class Node:

    def __init__(self, feature, df):
        self.feature = feature
        self.df = df
        self.left = None
        self.right = None
        self.feature_value = None
        self.label_value = None


class Tree:

    """
    确保数据框标签列列名为'label'
    """

    def __init__(self, df):
        self.df = df
        feature_name = self.get_feature(self.df)
        self.init_node = Node(feature_name, self.df)

    def get_feature(self, df):
        gini = {}
        for i in df.columns:
            if i != 'label':
                value_count_dict =  df[i].value_counts()
                sums = value_count_dict.values.sum()
                gini[i] = 1 - sum([(j * 1.0 / sums)**2 for j in value_count_dict.values])
        return max(gini, key=gini.get)

    @staticmethod
    def vote(df, columns_name, value):
        label_data = df.loc[df[columns_name] == value, 'label'].values
        return Counter(label_data).most_common()[0][0]

    def gen_tree(self, node):
        df = node.df
        feature_name = self.get_feature(df)
        feature_value_set = list(set(df[feature_name].values))
        if len(feature_value_set) > 2:
            raise ValueError
        elif len(feature_value_set) == 1:
            node.label_value = self.vote(df, feature_name, feature_value_set[0])
            return
        elif len(feature_value_set) == 2:
            left_node = Node(feature_name, df.loc[df[feature_name] == feature_value_set[0]])
            left_node.feature_value = feature_value_set[0]
            right_node = Node(feature_name, df.loc[df[feature_name] == feature_value_set[1]])
            right_node.feature_value = feature_value_set[1]
            node.left = left_node
            node.right = right_node
            self.gen_tree(left_node)
            self.gen_tree(right_node)

    def display_node(self, node, depth):
        if node.left == None:
            print "%slabel：%s" % ((depth - 1) *'\t|---' + '', node.label_value)
        else:
            print "%sfeature: %s, value: %s" % (depth * '\t' + '|---', node.left.feature, node.left.feature_value)
            self.display_node(node.left,depth+1)
            print "%sfeature: %s, value: %s" % (depth * '\t' +'|---', node.right.feature, node.right.feature_value)
            self.display_node(node.right,depth+1)


if __name__ == '__main__':
    data_set = [
        [1, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ]
    df = pd.DataFrame(data_set)
    df.columns = ['house', 'marriage', 'wage', 'label']
    tree = Tree(df)
    tree.gen_tree(tree.init_node)
    tree.display_node(tree.init_node, 0)
