from sklearn.datasets import load_boston

boston = load_boston()

from sklearn.cross_validation import train_test_split

import numpy as np;

X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)

print 'The max target value is: ', np.max(boston.target)
print 'The min target value is: ', np.min(boston.target)
print 'The average terget value is: ', np.mean(boston.target)

from sklearn.preprocessing import StandardScaler

ss_X = StandardScaler()
ss_y = StandardScaler()

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

from sklearn.svm import SVR

linear_svr = SVR(kernel = 'linear')

linear_svr.fit(X_train, y_train)

linear_svr_y_predict = linear_svr.predict(X_test)

poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print 'R-squared value of linear SVR is: ', linear_svr.score(X_test, y_test)
print 'The mean squared error of linear SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict))
print 'The mean absolute error of lin SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict))

print 'R-squared of ploy SVR is: ', poly_svr.score(X_test, y_test)
print 'the value of mean squared error of poly SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict))
print 'the value of mean ssbsolute error of poly SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict))

print 'R-squared of rbf SVR is: ', rbf_svr.score(X_test, y_test)
print 'the value of mean squared error of rbf SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict))
print 'the value of mean ssbsolute error of rbf SVR is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict))
