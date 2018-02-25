# -*- coding:utf-8 -*-
import numpy as np
import random
import copy
from simulated_annealing import exeTime
import matplotlib.pyplot as plt


class City():
    __slots__ = ("X", "Y")
    def __init__(self, x, y):
        self.X = x
        self.Y = y


class Graph:

    def __init__(self):
        self.city_list = []
        self.total_distance = 0

    @staticmethod
    def get_distance(city1, city2):
        return np.sqrt((city1.X - city2.X) ** 2 + (city1.Y - city2.Y) ** 2)

    def add_city(self, city):
        if isinstance(city, City):
            self.city_list.append(city)
        elif isinstance(city, list):
            self.city_list += city
        else:
            print 'Add City Wrong'

    def reset_distance(self):
        self.total_distance = 0

    def get_total_distance(self, sequence = None):
        if self.city_list == [] or len(self.city_list) < 2:
            print "请添加城市！"
        else:
            distance = 0
            if sequence == None:
                for i,city in enumerate(self.city_list[:-1]):
                    distance += self.get_distance(city, self.city_list[i+1])

                distance += self.get_distance(self.city_list[0], self.city_list[-1])

            elif sorted(sequence) == range(len(self.city_list)):
                self.reset_distance()
                for i,j in enumerate(sequence[:-1]):
                    distance += self.get_distance(self.city_list[j], self.city_list[sequence[i+1]])

                distance += self.get_distance(self.city_list[sequence[0]], self.city_list[sequence[-1]])

            else:
                print 'Wrong Sequence'
            return distance


def gen_new_sequence(sequence):
    sequence1 = copy.copy(sequence)
    swap_number1, swap_number2 = random.sample(sequence1, 2)
    sequence1[swap_number1], sequence1[swap_number2] = sequence1[swap_number2], sequence1[swap_number1]
    return copy.copy(sequence1)

#@exeTime
def get_shortest_distance(graph):
    T0 = 1000
    T_min = 1e-5
    delta = 0.9
    K = 10
    sequence = range(len(graph.city_list))
    distance = graph.get_total_distance(sequence)
    distance_list = []
    T = T0
    while T > T_min:
        for i in range(K):
            distance_list.append(distance)
            new_sequence = gen_new_sequence(sequence)

            new_distance = graph.get_total_distance(new_sequence)

            delta_E = new_distance - distance
            if delta_E < 0:
                distance = new_distance
                sequence = new_sequence
                break
            else:
                p_k = np.exp(- delta_E / T)
                if random.random() < p_k:
                    distance = new_distance
                    sequence = new_sequence
                    break
        T *= delta
    return sequence, distance, distance_list



if __name__ == '__main__':

    city_a = City(0, 0)
    city_b = City(0, 1)
    city_c = City(1, 0)
    city_d = City(1, 1)

    city_list = [
        City(0, 0),
        City(1, 0),
        City(2, 0),
        City(3, 0),
        City(4, 0),
        City(5, 2),
        City(0, 3),
        City(0, 4),
        City(0, 5),
        City(0, 6),
        City(1, 2),
        City(4, 3),
        City(50, 6),
        City(2, 3),
        City(1, 4),
        City(3, 16),
        City(3, 12),
        City(1, 12),
        City(12, 21),
        City(7, 8),
        City(5, 0),
        City(1, 9),
        City(2, 7),
        City(3, 7),
        City(10, 11),
        City(11, 1),
        City(17, 3),
        City(15, 3),
        City(22, 16),
        City(15, 1),
        City(8, 5),
        City(3, 1),
        City(2, 9),
        City(1, 9),
        City(9, 3),
        City(14, 1),
        City(12, 12),
    ]
    # 中国31省数据，最优值为15500以下
    chinese_province_list = [
        City(1304,2312),
        City(3639,1315),
        City(4177,2244),
        City(3712,1399),
        City(3488,1535),
        City(3326,1556),
        City(3238,1229),
        City(4196,1004),
        City(4312,790),
        City(4386,570),
        City(3007,1970),
        City(2562,1756),
        City(2788,1491),
        City(2381,1676),
        City(1332,695),
        City(3715,1678),
        City(3918,2179),
        City(4061,2370),
        City(3780,2212),
        City(3676,2578),
        City(4029,2838),
        City(4263,2931),
        City(3429,1908),
        City(3507,2367),
        City(3394,2643),
        City(3439,3201),
        City(2935,3240),
        City(3140,3550),
        City(2545,2357),
        City(2778,2826),
        City(2370,2975)
    ]

    graph = Graph()
    #graph.add_city([city_a, city_b, city_c, city_d])
    graph.add_city(chinese_province_list)
    result = get_shortest_distance(graph)
    plt.plot(result[2])
    plt.show()




    # result_list = []
    # for i in range(100):
    #     result = get_shortest_distance(graph)
    #     result_list.append(result[1])
    #     print result[1]
    # result_list.sort()
    # print result_list

