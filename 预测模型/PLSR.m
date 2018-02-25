clc;clear
Y=[ 0.1        0.5        0.7       
    0.2        0.6        0.4       
    0.3        0.7        0.5       
    0.4        0.6        0.3       
    0.5        0.8        0.2       
    0.6        0.3        0.5       
    0.4        0.7        0.6       
    0.3        0.5        0.7];
X=[0.2876        0.6173        0.9647        1.1936        1.0636        0.7332        0.5441        0.6247        0.7421        0.7052       
   0.2653        0.5167        0.8403        1.0435        1.008        0.7396        0.5344        0.5675        0.6312        0.5368       
   0.3833        0.7089        1.0544        1.2805        1.2524        0.8886        0.6596        0.6815        0.75        0.6671       
   0.3957        0.6853        0.9204        1.0648        1.0486        0.7999        0.5579        0.5381        0.5698        0.469       
   0.472         0.7413        1.0124        1.2202        1.2297        0.9699        0.6646        0.635        0.6254        0.4978       
   0.6268        0.9851        1.1633        1.1629        1.0128        0.7123        0.5161        0.482        0.5194        0.4909       
   0.4921        0.8723        1.2407        1.4583        1.3631        1.0073        0.7341        0.7032        0.8171        0.7228       
   0.4308        0.8232        1.146         1.309         1.1767        0.8207        0.5852        0.6604        0.7677        0.7237];
%X0=[0.4089        0.6996        0.8712        1.0159        0.9638        0.7115        0.5112        0.4722        0.5059        0.4343];     

[A,B,r,U,V,stats] = canoncorr(X,Y);
% A X变量数×典型变量个数，第i列表示自变量中的第i个典型变量里，X各个变量的系数，系数越大，表示影响越大
% B Y变量数×典型变量个数，第i列表示因变量中的第i个典型变量里，Y各个变量的系数
% r 典型变量个数，向量；表示自变量中第i个典型变量和因变量中第i个典型变量的相关系数，此时均为最大相关系数
% var( X * A(:,1)) = var( X * A(:,2)) = var( Y * B(:,1)) = 1
% U = (X - repmat(mean(X), size(X,1), 1)) * A
% V = (Y - repmat(mean(Y), size(Y,1), 1)) * B
% stats 统计参数，见https://cn.mathworks.com/help/stats/canoncorr.html



[XL,YL,XS,YS,BETA,PCTVAR,MSE] = plsregress(X,Y,7);
% X为8×10, Y为8×3
% 此时，获得7个主成分（默认获取样本数-1个主成分，根据解释方差累计图确认所需主成分），XL为10*7维矩阵
% 第i列表示第i个主成分，每一个变量对应的系数，共7个主成分
% XS为8*7维矩阵，第i列表示第i个主成分，每一个样本所对应的值
% 同理，YL，YS 如上
% BETA表示每个自变量对因变量的系数，注意包括了截距项，获取yhat时需要对X加上全为1的列，作为第一列
% PCTVAR表示每个成分的解释方差，第一行为X的，第二行为Y的
% MSE表示均方误差，第一行为X的，第二行为Y的
% https://cn.mathworks.com/help/stats/plsregress.html

plot(1:size(PCTVAR, 2),cumsum(100*PCTVAR(1,:)),'-bo');
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in x');

plot(1:size(PCTVAR, 2),cumsum(100*PCTVAR(2,:)),'-bo');
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in y');

Ytest = [ones(size(X,1),1) X]*BETA;
residuals = Y-Ytest;
stem(residuals)
xlabel('Observation');
ylabel('Residual');
