# -*- coding:utf-8 -*-
import numpy as np
import random
import math

"""
ref:
    http://www.cnblogs.com/heaad/archive/2010/12/23/1914725.html
    http://blog.csdn.net/czrzchao/article/details/52314455
    http://blog.csdn.net/u010451580/article/details/51178225

    求解tsp问题
    http://www.cnblogs.com/Key-Ky/p/3490991.html
"""

class Gene():

    __doc__ = "个体基因类，存储单个基因"

    def __init__(self, gene_length=10, float_length=4):
        self.gene_length = gene_length
        self.float_length = float_length        # 存储转换为十进制时小数位数，默认情况下，最大表示的值为
        self.bin_value = self.initial_gene()

    def initial_gene(self):
        """
        存储方式选择：
            字符串 :    '10001'        内存占用小(33+5)，但是需要转换整型计算，一定程度上拖慢速度
            列表*  :     [1,0,0,0,1]    内存占用大(104)，但是计算时间短
        :return:  列表
        """
        return [random.randint(0,1) for i in range(self.gene_length)]

    def bin2dec(self):
        """
        二进制转10进制，考虑小数情况
        """
        sums = 0
        for i,j in enumerate(self.bin_value):
            sums += int(j) * math.pow(2, self.gene_length - self.float_length - 1 - i)
        return sums

    @staticmethod
    def cross(gene1, gene2):
        assert gene1.gene_length == gene2.gene_length
        cut_start = random.randint(0, gene1.gene_length - 1)
        cut_end = random.randint(cut_start, gene1.gene_length)
        gene1.bin_value[cut_start:cut_end], gene2.bin_value[cut_start:cut_end] = \
            gene2.bin_value[cut_start:cut_end],gene1.bin_value[cut_start:cut_end]

    def mutation(self):
        i = random.randint(0, self.gene_length - 1)
        if self.bin_value[i] == 1:
            self.bin_value[i] = 0
        else:
            self.bin_value[i] = 1

    def get_fit_value(self, func):
        return func(self.bin2dec())


class GeneticAlgorithm:

    def __init__(self):
        self.population_size = 10
        self.gene_length = 10
        self.float_lenth = 3
        self.cross_prob = 0.6
        self.mutation_prob = 0.01
        self.func =  lambda x: (x - 20) ** 2 + 2 if x <= 50 else (x - 80) ** 2 + 30
        self.genes = [Gene() for i in range(self.population_size)]
        self.get_fit_value()

    def get_fit_value(self):
        self.fit_value = [i.get_fit_value(self.func) for i in self.genes]

    def get_best_gene(self):
        index = self.fit_value.index(min(self.fit_value))
        return self.genes[index], self.fit_value[index]

    def choose_gene(self, rand):
        """
        轮盘算法选择
        :param rand:    0-1随机数 
        :return:        选择的索引
        """
        _sum = 0.0
        total_fit_value = sum(self.fit_value)
        for i,j in enumerate(self.fit_value):
            _sum += j * 1.0 / total_fit_value
            if _sum > rand:
                return i

    def gene_pop(self):
        """
        基因淘汰
        :return: 
        """
        min_index = self.fit_value.index(min(self.fit_value))
        self.genes.pop(min_index)
        self.fit_value.pop(min_index)

    def begin(self):
        for i in range(1000):
            index1 = self.choose_gene(random.random())
            index2 = self.choose_gene(random.random())
            while index1 == index2:
                index2 = self.choose_gene(random.random())

            if random.random() < self.mutation_prob:
                self.genes[index1].mutation()
                self.genes[index2].mutation()

            if random.random() < self.cross_prob:
                Gene.cross(self.genes[index1], self.genes[index2])

            self.get_fit_value()
            # self.gene_pop()

            result = self.get_best_gene()
            print len(self.genes), result[0].bin2dec(), result[1]



if __name__ == '__main__':
    # gene1 = Gene()
    # print gene1.bin_value
    # print gene1.bin2dec()
    GA = GeneticAlgorithm()
    GA.begin()