% 优化约束条件，用在自带的优化工具箱中
function [cineq ,ceq ] = constraint(alpha)
cineq = []; % 不等式约束，写成小于等于0的形式的方程左侧，分好分割
ceq = norm(alpha) - 1; % 等式约束，写成等于0的形式的方程左侧，分号分割
end