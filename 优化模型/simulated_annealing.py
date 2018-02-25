# -*- coding:utf8 -*-
import numpy as np
import time
import random

"""
伪代码：
=================================================
随机生成初始解 x，对应目标函数为f(x)
开始温度 T0                  1e10
停止搜索温度 T_min      1e-8
温度下降速度 delta            0.9
每次迭代次数 K                100

T = T0
while T > T_threshold:
    for i in range(K):
        x' = gen(x)
        if f(x') < f(x):
            x = x'
        else:
            delta_E = f(x') - f(x)
            P_k = \frac{1}{1 + e^{-delta_E / T}}
            rand = random.random()
            if rand < P_k:
                x = x'
            else:
                pass
    T *= delta

=================================================
"""


def exeTime(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        print "%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__)
        print '-------------------  begin  ------------------------'
        back = func(*args, **args2)
        print '--------------------  end  -------------------------'
        print "%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__)
        print "%.8fs taken for {%s}" % (time.time() - t0, func.__name__)
        return back

    return newFunc


class SimulatedAnnealing:

    def __init__(self, func):
        self.T0 = 1000
        self.T_min = 1e-8
        self.delta = 0.99
        self.K = 10000
        self.x_range = (0, 100)
        self.func = lambda x:(x-20)**2 if x <= 50 else (x-80)**2 +30 #-(1.0 * x**4 - x**3 + x**2 - x)

    def gen_new_x(self, x_before, T):
        while 1:
            x_after = x_before + (random.random() * 2 - 1) * T
            if self.x_range[0] <= x_after <= self.x_range[1]:
                return x_after

    @exeTime
    def begin(self):
        x = random.randint(self.x_range[0], self.x_range[1])
        f = self.func(x)
        T = self.T0
        while T > self.T_min:
            for i in range(self.K):
                new_x = self.gen_new_x(x, T)
                f_x = self.func(new_x)
                delta_E = f_x - f
                #
                if delta_E < 0:
                    f = f_x
                    x = new_x
                    break
                else:
                    #p_k = 1.0 / (1 + np.exp(- delta_E / self.func(T)))
                    p_k = np.exp(- delta_E / T)
                    if random.random() < p_k:
                        f = f_x
                        x = new_x
                        break
            T *= self.delta

        return x


if __name__ == '__main__':
    sa = SimulatedAnnealing('')
    x = sa.begin()
    print x, sa.func(x)
