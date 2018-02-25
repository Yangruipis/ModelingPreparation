% this program use PPE method to judge the quality of a second-hand car

data= csvread('question4.csv',1,3);
this_size = size(data);
global n p standard_data;
p = this_size(2);
n = this_size(1);
standard_data = zeros(23,3);
for j = 1:p
   for i = 1:n
       if(ismember(j, [2,3]) == 1)
       standard_data(i,j) = (data(i,j) - min(data(:,j))) / ...
          (max(data(:,j)) - min(data(:,j)));
       else
       standard_data(i,j) = (max(data(:,j)) - data(i,j)) / ...
          (max(data(:,j)) - min(data(:,j)));
       end
   end
end

alpha = zeros(1,p);
for j = 1:p
    alpha(j) = 1/p;
end

%[a] = get_Q(alpha);
[value_list,best_a,b] = pso_optimal(100,3);

% load('auto.mat')
% this_size = size(auto_dataset);
% global n p standard_data;
% n = this_size(1);
% p = this_size(2);
% standard_data = zeros(74,11);
% 
% % premnmx() 归一化
% for j = 1:p
%    for i = 1:n
%        if(ismember(j, [1,2,4,5,6,7]) == 1)
%        standard_data(i,j) = (auto_dataset(i,j) - min(auto_dataset(:,j))) / ...
%           (max(auto_dataset(:,j)) - min(auto_dataset(:,j)));
%        else
%        standard_data(i,j) = (max(auto_dataset(:,j)) - auto_dataset(i,j)) / ...
%           (max(auto_dataset(:,j)) - min(auto_dataset(:,j)));
%        end
%    end
% end
% 
% % 初始化起点
% alpha = zeros(1,p);
% for j = 1:p
%     alpha(j) = 1/p;
% end
% 
% %[a] = get_Q(alpha);
% [best_a,b] = pso_optimal(100);
% 
% Z=zeros(n,1);
% for i=1:n
%     Z(i)=abs(sum(best_a.*standard_data(i,:)));
% end
% Z=abs(Z);
% 
% figure%投影散布图
% plot(abs(Z),'bd','LineWidth',1,'MarkerEdgeColor','k','MarkerFaceColor','b','MarkerSize',5);
% %axis([1,12,0,2.5]);%图形边界根据需要显示
% grid on
% xlabel('  ','FontName','TimesNewRoman','FontSize',12);
% ylabel('Projective Value','FontName','Times New Roman','Fontsize',12);
% figure
% [newZ,I]=sort(Z);
% plot(abs(newZ),'bd','LineWidth',1,'MarkerEdgeColor','k','MarkerFaceColor','b','MarkerSize',5);
% %axis([1,12,0,2.5]);%图形边界根据需要显示
% grid on
% xlabel('  ','FontName','TimesNewRoman','FontSize',12);
% ylabel('Projective Value','FontName','Times New Roman','Fontsize',12);
% 
% disp('最佳投影向量为')
% disp(best_a);