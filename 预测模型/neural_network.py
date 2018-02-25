# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing # 数据标准化

from pybrain.structure import *
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

"""
神经网络 对连续值进行预测
预安装pybrain: > pip install pybrain
ref:
    http://blog.csdn.net/u010900574/article/details/51290855
"""


def _generate_data():
    """
    生成数据集
    输入层为u(k-1) 和 y(k-1)，输出层为y(k)
    """
    # u = np.random.uniform(-1,1,200)
    # y=[]
    # former_y_value = 0
    # for i in np.arange(0,200):
    #     y.append(former_y_value)
    #     next_y_value = (29.0 / 40) * np.sin(
    #         (16.0 * u[i] + 8 * former_y_value) / (3.0 + 4.0 * (u[i] ** 2) + 4 * (former_y_value ** 2))) \
    #                    + (2.0 / 10) * u[i] + (2.0 / 10) * former_y_value
    #     former_y_value = next_y_value
    # return u,y
    u1 = np.random.uniform(-np.pi,np.pi,200)
    u2 = np.random.uniform(-1,1,200)
    y = np.zeros(200)
    for i in range(200):
        value = np.sin(u1[i]) + u2[i]
        y[i] =  value
    return u1, u2, y

def get_fnn():
    """
    创建层
        输入层:   2  units
        隐含层:   10 units
        输出层:   1  units
    """
    # createa neural network
    fnn = FeedForwardNetwork()
    # claim the layer
    inLayer = LinearLayer(2, name='inLayer')
    hiddenLayer0 = SigmoidLayer(10, name='hiddenLayer0')
    outLayer = LinearLayer(1, name='outLayer')
    # add three layers to the neural network
    fnn.addInputModule(inLayer)
    fnn.addModule(hiddenLayer0)
    fnn.addOutputModule(outLayer)
    # link three layers
    in_to_hidden0 = FullConnection(inLayer, hiddenLayer0)
    hidden0_to_out = FullConnection(hiddenLayer0, outLayer)
    # add the links to neural network
    fnn.addConnection(in_to_hidden0)
    fnn.addConnection(hidden0_to_out)
    # make neural network come into effect
    fnn.sortModules()

    return fnn

def get_train_data():
    # definite the dataset as two input , one output
    DS = SupervisedDataSet(2, 1)

    u1, u2, y = _generate_data()
    # add data element to the dataset
    for i in np.arange(199):
        DS.addSample([u1[i], u2[i]], [y[i + 1]])

    # you can get your input/output this way
    # X = DS['input']
    # Y = DS['target']

    # split the dataset into train dataset and test dataset
    dataTrain, dataTest = DS.splitWithProportion(0.8)

    return dataTrain, dataTest

def train_and_predict(fnn, dataTrain, dataTest):
    # train the NN
    # we use BP Algorithm
    # verbose = True means print th total error
    trainer = BackpropTrainer(fnn, dataTrain, verbose=True, learningrate=0.01)
    # set the epoch times to make the NN  fit
    trainer.trainUntilConvergence(maxEpochs=1000)

    xTest, yTest = dataTest['input'], dataTest['target']
    predict_resutl = []
    for i in np.arange(len(xTest)):
        predict_resutl.append(fnn.activate(xTest[i])[0])
    print(predict_resutl)

    plt.figure()
    plt.plot(np.arange(0, len(xTest)), predict_resutl, 'ro--', label='predict number')
    plt.plot(np.arange(0, len(xTest)), yTest, 'ko-', label='true number')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # for mod in fnn.modules:
    #     print ("Module:", mod.name)
    #     if mod.paramdim > 0:
    #         print ("--parameters:", mod.params)
    #     for conn in fnn.connections[mod]:
    #         print ("-connection to", conn.outmod.name)
    #         if conn.paramdim > 0:
    #             print ("- parameters", conn.params)
    #     if hasattr(fnn, "recurrentConns"):
    #         print ("Recurrent connections")
    #         for conn in fnn.recurrentConns:
    #             print ("-", conn.inmod.name, " to", conn.outmod.name)
    #             if conn.paramdim > 0:
    #                 print ("- parameters", conn.params)

def fnn_begin():
    fnn = get_fnn()
    dataTrain, dataTest = get_train_data()
    train_and_predict(fnn, dataTrain, dataTest)

class NeuralNetwork:

    def __init__(self, input_layer, hide_layer, output_layer, df):
        self.fnn = self.get_fnn(input_layer, hide_layer, output_layer)
        self.df = self.data_pre_handle(df)
        self.get_train_data(input_layer, output_layer)

    def data_pre_handle(self, df):
        """
        1. 剔除无分析价值列
        2. 缺失值补全
        3. 无效值剔除
        4. 分类变量编码
        5. 所有变量归一化
        
        """
        #df['类别'] = df['类别'].astype('category') # 节省内存开支
        df = df.dropna(axis=0)
        for column in df.columns:
            # 归一化
            df[column] = preprocessing.scale(df[column])
        return df

    def get_fnn(self, i, h, o):
        """
        创建层
            输入层:   i  units
            隐含层:   h  units
            输出层:   o  units
        """
        fnn = FeedForwardNetwork()

        inLayer = LinearLayer(i, name='inLayer')
        hiddenLayer0 = SigmoidLayer(h, name='hiddenLayer0')
        outLayer = LinearLayer(o, name='outLayer')

        fnn.addInputModule(inLayer)
        fnn.addModule(hiddenLayer0)
        fnn.addOutputModule(outLayer)

        in_to_hidden0 = FullConnection(inLayer, hiddenLayer0)
        hidden0_to_out = FullConnection(hiddenLayer0, outLayer)

        fnn.addConnection(in_to_hidden0)
        fnn.addConnection(hidden0_to_out)

        fnn.sortModules()
        return fnn

    def get_train_data(self, input_layer, output_layer):
        """
        输入数据为数据框，前input_layer列为输入数据，后output_layer列为输出数据
        """
        DS = SupervisedDataSet(input_layer, output_layer)

        for i in range(self.df.shape[0] - 1):
            DS.addSample(self.df.iloc[i, :input_layer].values, self.df.iloc[i+1, input_layer:].values)

        # 打乱顺序，取80%训练，20%测试
        # self.dataTrain, self.dataTest = DS.splitWithProportion(0.8)

        def split_by_part(DS, proportion=0.9):
            # 不随机抽取，而是取前80%的样本训练，后20%测试
            leftIndices = range(int(len(DS) * proportion))
            leftDs = DS.copy()
            leftDs.clear()
            rightDs = leftDs.copy()
            index = 0
            for sp in DS:
                if index in leftIndices:
                    leftDs.addSample(*sp)
                else:
                    rightDs.addSample(*sp)
                index += 1
            return leftDs, rightDs

        self.dataTrain, self.dataTest = split_by_part(DS, 0.99)

    def train(self, times = 1000):
        trainer = BackpropTrainer(self.fnn, self.dataTrain, verbose=True, learningrate=0.01)
        trainer.trainUntilConvergence(maxEpochs=times)

    def predict(self):
        xTest, yTest = self.dataTest['input'], self.dataTest['target']
        predict_resut = []
        for i in np.arange(len(xTest)):
            predict_resut.append(self.fnn.activate(xTest[i]))
        print(predict_resut)

        plt.figure()
        plt.plot(np.arange(0, len(xTest)), predict_resut, 'ro--', label='predict number')
        plt.plot(np.arange(0, len(xTest)), yTest, 'ko-', label='true number')
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def single_predict(self, x_array):
        return self.fnn.activate(x_array)

def Wind2Df(wind_data):
    df = pd.DataFrame(wind_data.Data).T
    df.columns = wind_data.Fields
    df.index = wind_data.Times
    return df


if __name__ == '__main__':
    # # fnn_begin()
    # df = pd.read_csv('dataset/auto.csv')
    # df = df.loc[:,[u'mpg', u'rep78', u'headroom', u'trunk', u'weight', u'length', u'turn', u'displacement', u'gear_ratio', u'price']]
    # nn = NeuralNetwork(9, 10, 1, df)
    # nn.train()
    # nn.predict()
    # print nn.single_predict(nn.df.ix[0].values[:9])
    # print nn.df.ix[0].values[-1]
    from WindPy import *
    import datetime
    w.start()
    df = Wind2Df(w.wst("IC1709.CFE",
                       "volume,amt,oi,bsize1,asize1,ask2,bid2,bsize2,asize2,bid3,ask3,bsize3,asize3,ask1,bid1,last",
                       "2017-08-22 09:00:00", "2017-08-22 14:45:05", ""))
    nn = NeuralNetwork(15, 15, 1, df)
    nn.train(100)
    nn.predict()





