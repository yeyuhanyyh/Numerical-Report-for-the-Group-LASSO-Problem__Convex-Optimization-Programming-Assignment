% ADMM求解器，求解原始问题（线性化）
function [y, num_iters, out] = gl_ADMM_primal(x0, A, b, mu, opts)
    % 获取A的大小
    [m, n] = size(A);
    l = size(b, 2);

    % 初始化迭代列表
    iters = [];
    % 初始化x和y
    x = x0;
    y = x0;
    % 初始化z
    z = zeros(n, l);

    % 设置参数t和eta
    t = 0.1;
    eta = 1;
    % 设置线性化选项
    linearize_x = false;
    linearize_y = true;
    
    % 计算逆矩阵
    inv1 = inv(t * eye(n) + A' * A);
    % 计算ATA和ATb
    ATA = A' * A;
    ATb = A' * b;

    % 开始迭代
    for it = 1:99999
        % 更新x
        if linearize_x
            x = x - eta * (ATA * x - ATb + z + t * (x - y));
        else
            x = inv1 * (t * y + ATb - z);
        end
        
        % 记录旧的y
        y0 = y;

        % 更新y
        if linearize_y
            y = prox(y - eta * (t * (y - x) - z), mu * eta);
        else
            y = prox(x + z / t, mu / t);
        end
        
        % 更新z
        z = z + t * (x - y);
        % 记录迭代次数和目标函数值
        iters = [iters; it, obj(A,b,mu,y)];

        % 计算原始饱和度和对偶饱和度
        primal_sat = norm(x - y);
        dual_sat = norm(y0 - y);
        % 检查是否收敛
        if primal_sat < 1e-4 && dual_sat < 1e-4
            break;
        end
    end

    % 返回迭代次数
    num_iters = size(iters, 1);
    % 返回迭代信息
    out.iters = iters;
    % 返回目标函数值
    out.fval = obj(A,b,mu,x);
% 在主函数中添加以下代码
end

% prox函数，用于计算proximal operator
function y_out = prox(y, mu)
    % 计算y的范数
    norm_y = vecnorm(y, 2, 2);
    % 计算proximal operator
    y_out = y .* max(0, norm_y - mu) ./ ((norm_y < mu * 0.1) + norm_y);
end

% 目标函数
function obj_val = obj(A,b,mu,X)
    % 计算目标函数值
    obj_val = 0.5 * norm(A * X - b, 'fro')^2 + mu * sum(vecnorm(X, 2, 2));
end
