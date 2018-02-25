%% 岭回归(Ridge Regression)  
 
% # 岭回归 （L2正则项最小二乘）
% 	- 有偏估计，但是在保证RSS足够小的情况下，使得参数更稳定
% 	- 在原先的最小二乘估计中加入扰动项（二阶正则项），使问题稳定有解
% 	- 岭回归针对样本没有办法提供给你足够的有效的信息的情况，此时OLS唯一存在的条件不满足，
% 		以损失部分信息、降低精度为代价获得回归系数更为符合实际、更可靠的回归方法，对病态数据的拟合要强于OLS
% 
% 	http://blog.csdn.net/google19890102/article/details/27228279
% 	http://f.dataguru.cn/thread-598486-1-1.html
    
%导入数据  
data = csvread('auto_1.csv', 1,0);  
[m,n] = size(data);  
  
dataX = data(:,1:10);%特征  
dataY = data(:,11);%标签  
  
%标准化  
yMeans = mean(dataY);  
for i = 1:m  
    yMat(i,:) = dataY(i,:)-yMeans;  
end  
  
xMeans = mean(dataX);  
xVars = var(dataX);  
for i = 1:m  
    xMat(i,:) = (dataX(i,:) - xMeans)./xVars;  
end  
  
% 运算30次  
testNum = 30;  
weights = zeros(testNum, n-1);  
for i = 1:testNum  
    w = ridgeRegression_func1(xMat, yMat, exp(i-10));  
    weights(i,:) = w';  
end  
  
% 画出随着参数lam  岭迹图
% λ的选择：一般通过观察，选取喇叭口附近的值，此时各β值已趋于稳定，但总的RSS又不是很大。
% 选择变量：删除那些β取值一直趋于0的变量。
hold on  
axis([-9 20 -1.0 2.5]);  
xlabel log(lam);  
ylabel weights;  
for i = 1:n-1  
    x = -9:20;  
    y(1,:) = weights(:,i)';  
    plot(x,y);  
end  

% 怎么看结果：
% 每一行对应一个lambda值，以及该lambda值下每个自变量的参数beta_i(lambda)