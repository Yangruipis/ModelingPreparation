# -*- coding:utf-8 -*-

"""
对所有预测模型的预测效果进行评估
ref:
    http://blog.csdn.net/sinat_26917383/article/details/75199996?locationNum=3&fps=1
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    https://www.zhihu.com/question/30643044
"""

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

TYPE_DISCRETE = 0       # 实际值与预测值均为离散
TYPE_DISCRETE_2 =1      # 实际值为离散，预测值为连续 logistic
TYPE_CONTINUE = 2       # 实际值与预测值均为连续


class Evaluate:

    def __init__(self, true_array, predict_array, pred_type = TYPE_DISCRETE):
        self.type = pred_type
        self.true_array = np.array(true_array)
        self.pred_array = np.array(predict_array)

    @property
    def accuracy(self):
        # 获取精确度
        # 采取宏平均 macro， 也可采用(None, ‘micro’, ‘macro’, ‘weighted’, ‘samples’)
        return metrics.precision_score(self.true_array, self.pred_array, average='macro')
    @property
    def recall(self):
        # 获取召回率
        return metrics.recall_score(self.true_array, self.pred_array, average='macro')

    @property
    def f1(self):
        # 获取F1值，即精确值和召回率的调和均值
        return metrics.f1_score(self.true_array, self.pred_array, average='weighted')

    @property
    def confusion_matrix(self):
        return metrics.confusion_matrix(self.true_array, self.pred_array)

    def confusion_matrix_plot(self, cmap=plt.cm.Blues):
        """Matplotlib绘制混淆矩阵图
        parameters
        ----------
            y_truth: 真实的y的值, 1d array
            y_predict: 预测的y的值, 1d array
            cmap: 画混淆矩阵图的配色风格, 使用cm.Blues，更多风格请参考官网
        """
        cm = metrics.confusion_matrix(self.true_array, self.pred_array)
        plt.matshow(cm, cmap=cmap)  # 混淆矩阵图
        plt.colorbar()  # 颜色标签

        for x in range(len(cm)):  # 数据标签
            for y in range(len(cm)):
                plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

        plt.ylabel('True label')  # 坐标轴标签
        plt.xlabel('Predicted label')  # 坐标轴标签
        plt.show()  # 显示作图结果

    @property
    def classify_report(self):
        return metrics.classification_report(self.true_array, self.pred_array)

    @property
    def kappa_score(self):
        # kappa score是一个介于(-1, 1)之间的数. score>0.8意味着好的分类；0或更低意味着不好
        return metrics.cohen_kappa_score(self.true_array, self.pred_array)

    @property
    def roc_score(self):
        return metrics.roc_auc_score(self.true_array, self.pred_array)

    def roc_plot(self, title='Receiver operating characteristic plot'):
        # 只针对二分了问题，如果是多分类，分别转换为二分类作图，即是第一类和不是第一类，是第二类和不是第二类等等
        fpr, tpr, _ = metrics.roc_curve(self.true_array, self.pred_array)
        plt.figure()
        # lw : line width
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % self.roc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()

    @property
    def hamming_distance(self):
        return metrics.hamming_loss(self.true_array, self.pred_array)

    @property
    def jaccard_distance(self):
        return metrics.jaccard_similarity_score(self.true_array, self.pred_array)
    
    @property
    def explained_variance(self):
        return metrics.explained_variance_score(self.true_array, self.pred_array)
    
    @property
    def mean_squared_error(self):
        return metrics.mean_squared_error(self.true_array, self.pred_array)
    
    @property
    def mean_absolute_error(self):
        return metrics.mean_absolute_error(self.true_array, self.pred_array)
    
    @property
    def median_absolute_error(self):
        return metrics.median_absolute_error(self.true_array, self.pred_array)

    @property
    def r_square(self):
        return metrics.r2_score(self.true_array, self.pred_array)

    def display(self):
        if self.type == TYPE_DISCRETE:
            print "accuracy : %s" % self.accuracy
            print "recall : %s" % self.recall
            print "F1 : %s" % self.f1
            print "confusion_matrix : \n %s" % self.confusion_matrix
            print "kappa : %s" % self.kappa_score
            print "ROC score : %s" % self.roc_score
            print "report : \n %s" % self.classify_report
            print "hamming loss : %s" % self.hamming_distance
            print "jaccard distance : %s" % self.jaccard_distance
            self.confusion_matrix_plot()
            self.roc_plot()
        elif self.type == TYPE_DISCRETE_2:
            print "ROC score : %s" % self.roc_score
            self.roc_plot()

        print "mean_squared_error : %s" % self.mean_squared_error
        print "mean_absolute_error : %s" % self.mean_absolute_error
        print "median_absolute_error : %s" % self.median_absolute_error
        print "explained_variance : %s" % self.explained_variance
        print "r_square : %s" % self.r_square


if __name__ == '__main__':
    true_y_0 = [1,1,0,1,0,1,1,1]
    pred_y_0 = [1,0,1,1,0,1,0,1]

    true_y_1 = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
    pred_y_1 = [1, 0.8, 0.2, 1.2, 0, 1.0, 0, 1.7, 2.1, 3.1]

    true_y_2 = [1, 1, 0.9, 1.1, 0.1, 1, 1, 0]
    pred_y_2 = [1, 0, 1, 1.2, 0, 1, 0, 1]

    eva_0 = Evaluate(true_y_0, pred_y_0, TYPE_DISCRETE)
    eva_1 = Evaluate(true_y_1, pred_y_1, TYPE_DISCRETE_2)
    eva_2 = Evaluate(true_y_2, pred_y_2, TYPE_CONTINUE)

    eva_0.display()
    eva_1.display()
    eva_2.display()

















