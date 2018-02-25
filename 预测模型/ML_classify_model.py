# -*- coding:utf-8 -*-
from sklearn import cross_validation
import numpy as np
import pandas as pd

class myclassify():
    def __init__(self, train_x, train_y):
        self.x = train_x
        self.y = train_y
        self.cv_time = 10

    def knn(self, k=3):
        from sklearn import neighbors
        knn_model = neighbors.KNeighborsClassifier(n_neighbors=k)
        scores = cross_validation.cross_val_score(knn_model, self.x, self.y, cv=self.cv_time)
        knn_model.fit(self.x, self.y)
        return np.mean(scores), knn_model

    def logistic(self):
        from sklearn.linear_model import LogisticRegression
        logit_model = LogisticRegression()
        scores = cross_validation.cross_val_score(logit_model, self.x, self.y, cv=self.cv_time)
        logit_model.fit(self.x, self.y)
        return np.mean(scores), logit_model

    def decision_tree(self):
        from sklearn import tree
        dt_model = tree.DecisionTreeClassifier(criterion='entropy')
        scores = cross_validation.cross_val_score(dt_model, self.x, self.y, cv=self.cv_time)
        dt_model.fit(self.x, self.y)
        return np.mean(scores), dt_model

    def naive_bayes(self):
        from sklearn.naive_bayes import MultinomialNB
        nb_model = MultinomialNB()
        scores = cross_validation.cross_val_score(nb_model, self.x, self.y, cv=self.cv_time)
        nb_model.fit(self.x, self.y)
        return np.mean(scores), nb_model

    def svm(self):
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', probability=True)
        scores = cross_validation.cross_val_score(model, self.x, self.y, cv=self.cv_time)
        model.fit(self.x, self.y)
        return np.mean(scores), model

    def svm_cv(self):
        from sklearn.grid_search import GridSearchCV
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', probability=True)
        param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
        grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
        grid_search.fit(self.x, self.y)
        best_parameters = grid_search.best_estimator_.get_params()
        for para, val in list(best_parameters.items()):
            print(para, val)
        model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
        scores = cross_validation.cross_val_score(model, self.x, self.y, cv=self.cv_time)
        model.fit(self.x, self.y)
        return scores, model

if __name__ == '__main__':
    df = pd.read_csv("../dataset/auto_1.csv")
    df = df.dropna(axis=0)
    mc = myclassify(df.iloc[:, 0:10], df.iloc[:,-1])
    #scores, model = mc.knn(3)
    #scores, model = mc.svm()
    scores, model = mc.svm_cv()
    predict_num = -3
    print scores,model.predict(df.iloc[predict_num,0:10].values.T)[0],df.iloc[predict_num,-1]


