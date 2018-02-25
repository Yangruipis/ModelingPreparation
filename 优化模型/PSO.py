# -*- coding:utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt

"""
优化算法之粒子群算法：

对于优化函数f(x_1, x_2, ..., x_n)求最小值

给定m个粒子，每个粒子有n个维度
针对第i个粒子(0<i<=m)，X_i = (x_1, x_2, x_3, ..., x_n), V_i = (v_1, v_2, v_3, ..., v_n)

每个粒子通过以下方法更新自己的速度和位置:

(1) V_i^{t+1} = wV_i^t + c_1 r_1 (p_i - X_i^t) + c_2 r_2 (p_g - X_i^t)
(2) X_i^{t+1} = X_i^t + V_i^{t+1}

其中：
    V_i^t:  当前速度
    X_i^t:  当前粒子位置
    w    :  惯性权重，0.8左右
    p_i  :  此粒子历史上最好的位置
    p_g  :  所有粒子历史上最好的位置
    c_1  :  学习因子
    c_2  :  学习因子，通常，c_1 = c_2 = 2
    r_1  :  [0,1]之间的随机数
    r_2  :  [0,1]之间的随机数


ref: http://blog.csdn.net/chen_jp/article/details/7947059
     http://blog.csdn.net/kunshanyuz/article/details/63683145

"""



class PSO():

    def __init__(self, particle_number=10, variable_number=1):
        self.w = 0.8
        self.c1 = self.c2 = 2
        self.r1 = 0.25
        self.r2 = 0.75
        self.particle_number = particle_number                                          # 粒子数
        self.variable_number = variable_number                                          # 变量个数
        self.X = np.zeros((particle_number, variable_number))                           # 存储每个粒子位置信息
        self.V = np.zeros((particle_number, variable_number))                           # 存储每个粒子速度信息
        self.particle_optimal_position = np.zeros((particle_number, variable_number))   # 存储每个粒子最优位置信息
        self.optimal_position = np.zeros((1, variable_number))                          # 存储全局最优位置
        self.optimal_value = 1e10                                                       # 存储全局最优值
        self.x_range = (0, 100)                                                         # 存储所有x的取值范围
        self.initial_particle()

    def func(self, array):
        """
        需要求解的目标函数
        :param array:   单个粒子的位置
        :return:        目标函数值
        """
        function = lambda x: (x - 20) ** 2 + 2 if x <= 50 else (x - 80) ** 2 + 30  # -(1.0 * x**4 - x**3 + x**2 - x)
        return function(array[0])

    def initial_particle(self):
        """
        初始化粒子位置与最优信息
        """
        for i in range(self.particle_number):
            for j in range(self.variable_number):
                self.X[i][j] = self.x_range[0] + (self.x_range[1] - self.x_range[0]) * random.random()
                self.V[i][j] = 0
            self.particle_optimal_position[i] = self.X[i]
            value = self.func(self.X[i])
            if value < self.optimal_value:
                self.optimal_position = self.X[i]
                self.optimal_value = value

    def update_particle(self):
        for i in range(self.particle_number):
            self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.particle_optimal_position[i] - self.X[i]) \
                        + self.c2 * self.r2 * (self.optimal_position[0] - self.X[i])
            self.X[i] = self.X[i] + self.V[i]

            if self.x_range[0] <= self.X[i].any() <= self.x_range[1]:
                value_before = self.func(self.particle_optimal_position[i])
                value_now = self.func(self.X[i])
                if value_now < value_before:
                    self.particle_optimal_position[i] = self.X[i]

                if value_now < self.optimal_value:
                    self.optimal_position = self.X[i]
                    self.optimal_value = value_now


    def pso_begin(self):
        for i in range(1000):
            self.update_particle()
            print "最优粒子位置：%s, 最优值：%s" % (self.optimal_position, self.optimal_value)


if __name__ == '__main__':
    pso = PSO()
    pso.pso_begin()






































#
# # ----------------------PSO参数设置---------------------------------
# class PSO():
#     def __init__(self, pN, dim, max_iter):
#         self.w = 0.8
#         self.c1 = 2
#         self.c2 = 2
#         self.r1 = 0.6
#         self.r2 = 0.3
#         self.pN = pN  # 粒子数量
#         self.dim = dim  # 搜索维度
#         self.max_iter = max_iter  # 迭代次数
#         self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置和速度
#         self.V = np.zeros((self.pN, self.dim))
#         self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置和全局最佳位置
#         self.gbest = np.zeros((1, self.dim))
#         self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
#         self.fit = 1e10  # 全局最佳适应值
#
# # ---------------------目标函数Sphere函数-----------------------------
#     def function(self, x):
#         sum = 0
#         length = len(x)
#         x = x ** 2
#         for i in range(length):
#             sum += x[i]
#         return sum
#
# # ---------------------初始化种群----------------------------------
#
#
# def init_Population(self):
#     for i in range(self.pN):
#         for j in range(self.dim):
#             self.X[i][j] = random.uniform(0, 1)
#             self.V[i][j] = random.uniform(0, 1)
#         self.pbest[i] = self.X[i]
#         tmp = self.function(self.X[i])
#         self.p_fit[i] = tmp
#         if (tmp < self.fit):
#             self.fit = tmp
#             self.gbest = self.X[i]
#
# # ----------------------更新粒子位置----------------------------------
#
#
# def iterator(self):
#     fitness = []
#     for t in range(self.max_iter):
#         for i in range(self.pN):  # 更新gbest\pbest
#             temp = self.function(self.X[i])
#             if (temp < self.p_fit[i]):  # 更新个体最优
#                 self.p_fit[i] = temp
#                 self.pbest[i] = self.X[i]
#                 if (self.p_fit[i] < self.fit):  # 更新全局最优
#                     self.gbest = self.X[i]
#                     self.fit = self.p_fit[i]
#         for i in range(self.pN):
#             self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) \
#                         + self.c2 * self.r2 * (self.gbest - self.X[i])
#             self.X[i] = self.X[i] + self.V[i]
#         fitness.append(self.fit)
#         print(self.fit)  # 输出最优值
#     return fitness
#
# # ----------------------程序执行-----------------------
# my_pso = PSO(pN=30, dim=5, max_iter=100)
# my_pso.init_Population()
# fitness = my_pso.iterator()
# # -------------------画图--------------------
# plt.figure(1)
# plt.title("Figure1")
# plt.xlabel("iterators", size=14)
# plt.ylabel("fitness", size=14)
# t = np.array([t for t in range(0, 100)])
# fitness = np.array(fitness)
# plt.plot(t, fitness, color='b', linewidth=3)
# plt.show()