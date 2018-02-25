% 优化目标函数
function [result] = get_Q(alpha)
global n p standard_data;
R = 1;
z = zeros(n,1);
for i = 1:n
    sum = 0;
    for j = 1:p
        sum = sum + alpha(j) * standard_data(i,j);
    end
    z(i) = sum;
end
S_alpha = std(z);
sum_d = 0;
for i = 1:n
    for j = 1:p
        u = 0;
        temp = R - abs(z(i) - z(j));
        if(temp>=0)
            u = 1;
        end
        sum_d = sum_d + temp * u;
    end
end
result =- S_alpha * sum_d;
end