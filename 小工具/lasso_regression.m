% Lasso 回归
% 
% # lasso	（L1正则项最小二乘）
% 	lasso用于变量筛选
% 	
% 	http://blog.csdn.net/sinat_26917383/article/details/52092040 # 正则项，lasso和ridge区别
% 	matlab实现：	http://cn.mathworks.com/help/stats/lasso.html
% 	

% lasso
data = csvread('auto_1.csv',1,0);
X = data(:,1:10);
Y = data(:, 11);
weight = lasso(X,Y);

% plot
hold on  
axis([0 100 -0.6 0.1]);  
xlabel log(lam);  
ylabel weights;  
y = zeros(1,100);
for i = 1:10
    x = 0:99;  
    y(1,:) = weight(i,:);  
    plot(x,y);  
    %legend(int2str( i));
end  
legend('1','2','3','4','5','6','7','8','9','10');
hold off

% 结果分析：
% 默认输出lambda=0:99时对应的参数，返回一个100行n列（n为自变量数）的矩阵
% lasso结果用以筛选变量，无关变量的系数会趋近与0，相关变量则不会

% 含参数的lasso
% 1. lasso & ridge
% 采用ridge回归, lasso占比1%, 输出lambda = 0:19
lasso(data(:,1:10), data(:,11), 'Alpha', 0.01, 'NumLambda', 20)

