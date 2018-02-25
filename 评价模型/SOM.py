# -*- coding:utf-8 -*-
# Self Organizing Maps for clustering
"""
相关文档
theory:
    - http://www.cnblogs.com/sylvanas2012/p/5117056.html
    - http://www.68dl.com/research/2014/0922/9129.html
matlab:
    - http://blog.sina.com.cn/s/blog_906d892d0102vxfv.html
    - http://blog.csdn.net/bwangk/article/details/53300622
    - https://cn.mathworks.com/help/nnet/ug/cluster-with-self-organizing-map-neural-network.html
python:
    - http://blog.csdn.net/chenge_j/article/details/72537568    个人实现
    - https://github.com/sevamoo/SOMPY                          官方包
"""

import numpy as np
from matplotlib import pyplot as plt
from sompy.sompy import SOMFactory
from sklearn.datasets import fetch_california_housing
import pandas as pd
from collections import Counter


class MySOM:
    def __init__(self, df, mapsize, initialization = 'random'):
        """
        
        :param df:              数据框 
        :param mapsize:         输出层维度，一般为二维，输入(20,20)的形式
        :param initialization:  "PCA" 或 "random"，初始化权重的方法
                - PCA是以变量的主成分值作为权重，见sompy.codebool.pca_linear_initialization
                - random是以随机数进行初始化
        """
        self.data = np.array(df)
        self.sm = SOMFactory().build(self.data, mapsize=mapsize, initialization=initialization, component_names=df.columns)
        self.train()

    def train(self):
        self.sm.train(n_job=1,verbose=False, train_rough_len=2, train_finetune_len=5)

    def print_error(self):
        topographic_error = self.sm.calculate_topographic_error()
        quantization_error = np.mean(self.sm._bmu[1])
        print ("Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error))

    def draw_input_weights(self):
        from sompy.visualization.mapview import View2D
        view2D = View2D(10, 10, "rand data", text_size=10)
        view2D.show(self.sm, col_sz=4, which_dim="all", desnormalize=True)
        plt.show()

    def draw_hit_map(self):
        from sompy.visualization.bmuhits import BmuHitsView
        vhts = BmuHitsView(4, 4, "Hits Map", text_size=12)
        vhts.show(self.sm, anotate=True, onlyzeros=False, labelsize=12, cmap="Greys", logaritmic=False)
        plt.show()

    def draw_cluster_map(self):
        from sompy.visualization.hitmap import HitMapView
        hits = HitMapView(20, 20, "Clustering", text_size=12)
        hits.show(self.sm)
        plt.show()

    def cluster(self, n):
        self.sm.cluster(n)

    def get_cluster_label(self):
        # 长度等于mapsize[0] * mapsize[1]
        return self.sm.cluster_labels

    def get_neurons(self):
        """
        获取原数据的每个样本对应的神经元，原包并未提供此方法，所以自己动手
        :return: array, length = self.df.shape[0]
        """
        return self.sm._bmu[0]

    def get_label(self):
        """
        获取原数据的每个样本对应的分类标签，原包并未提供此方法，所以自己动手
        :return: array, length = self.df.shape[0]
        """
        neurons_label_dict = {i:j for i,j in enumerate(self.sm.cluster_labels)}
        return np.array([neurons_label_dict[i] for i in self.sm._bmu[0]])

    def predict(self, x):
        """
        以label作为y，采取各种机器学习算法
        :param x: 
        :return: 
        """
        pass

if __name__ == '__main__':
    data = fetch_california_housing()
    descr = data.DESCR
    names = data.feature_names+["HouseValue"]
    data = np.column_stack([data.data, data.target])
    df = pd.DataFrame(data)
    df.columns = names

    my_som = MySOM(df, (20,20))
    my_som.draw_input_weights()
    my_som.draw_hit_map()

    my_som.cluster(5)
    my_som.draw_cluster_map()
    print my_som.get_label()[:10]
    print Counter(my_som.get_label())

    my_som.predict(np.array(df.iloc[0]))