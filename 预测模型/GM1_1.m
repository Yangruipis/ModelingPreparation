%%%% GM(1,1)预测模型

input = [132,92,118,130,187,207,213,284,301,333];
predict_times = 10;

sum_input = cumsum(input);

B = ones(length(input)-1,2);
for i = 1:length(input)-1
    B(i,1) = -(sum_input(i) + sum_input(i+1)) / 2.0;
end
Y = input(2:end);

a_hat = inv(B' * B) * B' * Y';
a = a_hat(1);
u = a_hat(2);

result_length = length(input) + predict_times;
sum_result = zeros(result_length, 1);
result = zeros(result_length, 1);
for i = 1:result_length
    sum_result(i) = (input(1) - u / a) * exp(-a * (i - 1)) + u / a;
    if(i == 1)
        result(i) = sum_result(i);
    else
        result(i) = sum_result(i) - sum_result(i-1);
    end
end
x_1 = 1:length(input);
x_2 = 1:result_length;

plot(x_1, input, '.b', x_2, result, 'ro--')
legend('Acture Value', 'Predict Value')
title('GM(1,1) Predict Result')



