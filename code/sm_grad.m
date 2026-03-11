% 定义函数，输入包括当前解x，数据矩阵A，观测向量b，正则化参数mu和光滑化参数sm_alpha
function sg = sm_grad(x, A, b, mu, sm_alpha)
    grad_fx = A.'* (A * x - b); % 计算目标函数关于x的梯度
    row_norm = sqrt(sum(x.^2, 2)); % 计算x的每一行的2范数
    row_norm(row_norm < sm_alpha) = sm_alpha; % 对于2范数小于光滑化参数的行，将其2范数设为平滑参数
    row_norm = row_norm.^-1; % 计算每个2范数的倒数
    subgrad_reg = diag(row_norm) * x; % 计算正则项的次梯度
    sg = grad_fx + mu * subgrad_reg; % 计算总的次梯度
end


