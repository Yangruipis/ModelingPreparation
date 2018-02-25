# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

class Cluster:

    def __init__(self, df):
        from scipy.cluster.vq import whiten
        self.df = df
        self.data = whiten(df)
        self.sample_names = np.array(df.index)

    def K_means(self, K, axis=0):
        from scipy.cluster.vq import kmeans, vq
        # k-means最后输出的结果其实是两维的,第一维是聚类中心,第二维是损失distortion
        if axis == 0:
            # 此时对样本聚类
            centroid, distortion = kmeans(self.data, K)
            # 使用vq函数根据聚类中心对所有数据进行分类,vq的输出也是两维的,[0]表示的是所有数据的label
            label, distortion = vq(self.data, centroid)
        else:
            # 此时对变量聚类
            centroid, distortion = kmeans(self.data.T, K)
            label, distortion = vq(self.data.T, centroid)
        return label

    def hierarchical(self):
        import scipy.cluster.hierarchy as sch
        # 生成点与点之间的距离矩阵,这里用的欧氏距离:
        disMat = sch.distance.pdist(self.data, 'euclidean')
        # 进行层次聚类:
        Z = sch.linkage(disMat, method='average')
        self.hierarchial_plot(Z)
        # 根据linkage matrix Z得到聚类结果:
        cluster = sch.fcluster(Z, 1, 'inconsistent')
        return cluster

    def hierarchial_plot(self, Z):
        import scipy.cluster.hierarchy as sch
        # 将层级聚类结果以树状图表示出来，其中labels为每个样本的名称数组,应该为self.sample_names
        sch.dendrogram(Z, labels=self.sample_names, orientation='right')
        plt.tick_params(
            axis='x',  # 使用 x 坐标轴
            which='both',  # 同时使用主刻度标签（major ticks）和次刻度标签（minor ticks）
            bottom='off',  # 取消底部边缘（bottom edge）标签
            top='off',  # 取消顶部边缘（top edge）标签
            labelbottom='off')
        plt.tight_layout()  # 展示紧凑的绘图布局
        plt.show()
        # plt.savefig('plot_dendrogram.png')

    def cluster_plot(self, label):
        # 聚类结果适合在二维数据中进行可视化，而面对多维情况，采取主成分分析进行降唯
        pca_result = self._pca()
        color = ['r', 'y', 'k', 'g', 'm'] * 10
        for i in range(max(label)+1):
            idx = np.where(label==i)
            plt.scatter(pca_result[idx, 0], pca_result[idx, 1], marker='o',label = str(i), color=color[i])
        plt.legend([u"Class: "+ str(i) for i in range(max(label) + 1)])
        plt.show()

    def _pca(self):
        pca = PCA(n_components=2) # ='mle' 时自动判断需要保留几个主成分，在这里因为需要做图，所以保留前两个
        pca.fit(self.data)
        print "variance_ratio:", pca.explained_variance_ratio_
        return pca.transform(self.data)

    def auto_cluster(self):
        # 先层次聚类，获取分类数，再根据类别进行K均值聚类
        hierarchical_cluster = self.hierarchical()
        K = max(hierarchical_cluster)
        labels = self.K_means(K)
        self.cluster_plot(labels)


if __name__ == '__main__':
    df = pd.read_csv("/home/ray/Documents/suibe/2017/建模/Modeling_Preparation/dataset/auto_1.csv")
    df = df.dropna(axis=0)
    clu = Cluster(df)

    label = clu.K_means(4)
    clu.cluster_plot(label)

    label2 = clu.hierarchical()
    clu.cluster_plot(label2)
