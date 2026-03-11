% 定义函数，输入包括当前解x，数据矩阵A，观测向量b，正则化参数mu和阈值e
function sg = subgrad(x, A, b, mu, e)
    grad_fx = A.'* (A * x - b); % 计算目标函数关于x的梯度
    row_norm = sqrt(sum(x.^2, 2)); % 计算x的每一行的2范数
    row_norm(row_norm < e) = inf; % 对于2范数小于阈值e的行，将其2范数设为无穷大
    row_norm = row_norm.^-1; % 计算每个2范数的倒数
    subgrad_reg = diag(row_norm) * x; % 计算正则项的次梯度
    sg = grad_fx + mu * subgrad_reg; % 计算总的次梯度
end

