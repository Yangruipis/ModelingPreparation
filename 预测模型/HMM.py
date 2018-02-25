# -*- coding:utf-8 -*-

"""
Theory:

    通过识别隐含状态，以及计算隐含状态到观测状态的概率，实现对未来隐含状态的预测（
        方法1. 已知当期的隐状态，推断下一期隐状态概率，以及各自的观测分布情况，进行预测
        方法2. 根据当期t_1的观测，寻找与当期最相似的时期t_2，类比t_2下一期的观测值金预测
    ）

    模型参数：隐含状态转移概率矩阵、隐状态->观测转移概率(emission matrix, 混淆矩阵）、初始隐状态概率

    根据观测分类：
        - MultinomialHMM    观测值离散的HMM
        - GaussianHMM       观测值连续的HMM，当观测为一维时，假定为正态分布；当观测为n维时，为n维联合正态分布
        - GMMHMM            同样为连续观测，运用混合正态分布

    根据问题分类：
        1. 已知整个模型（包括转移概率矩阵、混淆矩阵），根据观测值序列，计算该序列产生的概率如何
        2. 已知整个模型（包括转移概率矩阵、混淆矩阵），根据观测值序列，推断这段时间的隐含状态
        3. 模型未知，只知道观测值序列，求解整个模型，计算两个概率矩阵（或者是概率分布，连续情况），以及初始隐含状态概率（分布）

    对应求解方法：
        1. 前向、后向算法
        2. Viterbi Algo，维特比算法
        3. Baum-Welch Algo，鲍姆-韦尔奇算法
    ref：
        https://www.zhihu.com/question/20962240

python:
    ref:
        http://www.cnblogs.com/pinard/p/7001397.html
        https://uqer.io/community/share/56ec30bf228e5b887be50b35    # 量化
        http://blog.csdn.net/baskbeast/article/details/51218777     # 量化

"""

import hmmlearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def MyMultinomialHMM():
    from hmmlearn import hmm

    # 离散观测情况
    states = ["box 1", "box 2", "box3"]
    n_states = len(states)

    observations = ["red", "white"]
    n_observations = len(observations)

    start_probability = np.array([0.2, 0.4, 0.4])

    transition_probability = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ])

    emission_probability = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]
    ])

    model = hmm.MultinomialHMM(n_components=n_states)
    model.startprob_ = start_probability
    model.transmat_ = transition_probability
    model.emissionprob_ = emission_probability

    # question 2
    seen = np.array([[0, 1, 0, 1, 0, 0, 1]]).T  # 观测序列
    logprob, box = model.decode(seen, algorithm="viterbi")
    print "The ball picked:", ", ".join(map(lambda x: observations[x], seen.T.reshape(7)))
    print "The hidden box", ", ".join(map(lambda x: states[x], box))

    box2 = model.predict(seen)
    print "The ball picked:", ", ".join(map(lambda x: observations[x], seen.T.reshape(7)))
    print "The hidden box", ", ".join(map(lambda x: states[x], box2))

    # question 1
    print np.exp(model.score(seen))

    # question 3

    # states = ["box 1", "box 2", "box3"]
    n_states = 3  # 参数 1
    X2 = np.array([[0, 1, 0, 1], [0, 0, 0, 1], [1, 0, 1, 1]])  # 参数 2

    model2 = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.01)
    model2.fit(X2)
    for i in range(10):
        # 由于鲍姆-韦尔奇算法是基于EM算法的近似算法，所以我们需要多跑几次，选择X2概率最大的作为模型估计结果
        model2.fit(X2)
        print model2.startprob_
        print model2.transmat_
        print model2.emissionprob_
        print np.exp(model2.score(X2))
    print model2.sample(10)
    print model2.predict(X2.reshape([3, 4, 1])[1])

def MyGaussianHMM():
    from hmmlearn.hmm import GaussianHMM
    df = pd.read_csv("/home/ray/Documents/suibe/2017/建模/Modeling_Preparation/dataset/SZIndex.csv", header=-1)
    df.head()
    X = np.array(df.iloc[:, 0:5])

    # 一、未知模型情况下，解决问题3
    model = GaussianHMM(n_components=6, covariance_type="diag", n_iter=1000)  # 方差矩阵为对角阵
    """
    参数解释：
    covariance_type:
        "spherical"     ：主对角元素均为1，其余元素为0，独立同分布  (数据不足时，难以进行参数估计)
        "diag"          ：主对角元素不为0，其余为0               (一般情况，折中)
        "full"          ：所有元素均不为0                      (数据足够进行参数估计时)
    """
    model.fit(X)
    print "隐含状态为: ", model.predict(X)  # 列出每一天的隐含状态
    print "特征数目 %s" % model.n_features
    print "隐状态数目 %s" % model.n_components
    print "起始概率 :", model.startprob_
    print "隐状态转移矩阵", model.transmat_
    ## 每个隐含层对应的特征概率空间假设为正态分布，则可以得到一个model.n_components行model.n_features列的均值矩阵
    print "混淆矩阵：均值部分", model.means_
    print "混淆矩阵：方差部分", model.covars_

    ## 绘图
    hidden_states = model.predict(X)
    tradeDate = df.iloc[:, 5].values
    closeIndex = df.iloc[:, 6].values
    plt.figure(figsize=(15, 8))
    for i in range(model.n_components):
        idx = (hidden_states == i)
        plt.plot_date(pd.to_datetime(tradeDate[idx]), closeIndex[idx], '.', label='%dth hidden state' % i, lw=1)
        plt.legend()
        plt.grid(1)
    plt.show()

    # 二、已知模型情况下，解决问题1,2

    ## 沿用上述模型
    ### 问题1
    print "某天出现该观测的概率为： %s" % np.exp(model.score(X[0]))
    ### 问题2
    log_prob, state = model.decode(X[:10], algorithm="viterbi")
    print "只根据前十天，推断出最有可能的隐含状态序列为：", state

    ## 自己输入模型参数
    ### 一个2特征，4隐状态情况
    startprob = np.array([0.6, 0.3, 0.1, 0.0])
    # The transition matrix, note that there are no transitions possible
    # between component 1 and 3
    transmat = np.array([[0.7, 0.2, 0.0, 0.1],
                         [0.3, 0.5, 0.2, 0.0],
                         [0.0, 0.3, 0.5, 0.2],
                         [0.2, 0.0, 0.2, 0.6]])
    # The means of each component
    means = np.array([[0.0, 0.0],
                      [0.0, 11.0],
                      [9.0, 10.0],
                      [11.0, -1.0]])
    # The covariance of each component
    covars = .5 * np.tile(np.identity(2), (4, 1, 1))
    model2 = GaussianHMM(n_components=4, covariance_type="full", n_iter=1000)
    model2.startprob_ = startprob
    model2.transmat_ = transmat
    model2.means_ = means
    model2.covars_ = covars

if __name__ == '__main__':
    MyGaussianHMM()
    pass
