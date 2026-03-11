function [la, num_iters, out] = gl_ALM_dual(x0, A, b, mu, opts)
    [m, n] = size(A); % 获取矩阵A的大小
    l = size(b, 2); % 获取向量b的长度
    MAX_ITER = 9999; % 设置最大迭代次数
    th_converge = 1e-12; % 设置收敛阈值
    MAX_INNER_ITER1 = 1; % 设置内部循环的最大迭代次数
    iters_primal = []; % 初始化原始问题的迭代次数列表
    iters_dual = []; % 初始化对偶问题的迭代次数列表
    scale = 2^16; % 初始化缩放因子
    la = -x0; % 初始化拉格朗日乘子
    t = 1e-3; % 初始化子问题迭代步长
    inv1 = inv(eye(m) + t * A * A'); % 计算逆矩阵
    v = zeros(n, l); % 初始化向量v
    scale_mu = scale * mu; % 计算缩放后的mu
    for it = 1:MAX_ITER % 开始外部循环
        if rem(it,100)==0 % 每100次迭代
            scale = max(scale / 4, 1); % 缩小缩放因子
            scale_mu = scale * mu; % 更新缩放后的mu
        end
        for it2 = 1:MAX_INNER_ITER1 % 开始内部循环
            z = inv1 * (t * A * v - A * la - b); % 计算向量z
            v0 = v; % 保存当前的v值
            v = project(A' * z + la / t, scale_mu); % 更新v的值          
            inner_gap = norm(v - v0, 'fro'); % 计算v和v0之间的Frobenius范数
            if inner_gap < th_converge % 检查是否满足内部循环的收敛条件
                break; % 如果满足，跳出内部循环
            end
        end
        
        la = la + t * (A' * z - v); % 更新拉格朗日乘子
        %la(abs(la)<1e-5) = 0;
        iters_primal = [iters_primal; it, primal_obj(A,b,scale_mu,-la)]; % 计算原始问题的目标函数值，并添加到列表中
        iters_dual = [iters_dual; it, dual_obj(b,z)]; % 计算对偶问题的目标函数值，并添加到列表中

        dual_sat = norm(A' * z - v); % 计算对偶满足度
        if dual_sat < th_converge % 检查是否满足外部循环的收敛条件
            break; % 如果满足，跳出外部循环
        end
    end
    primal_obj_value = iters_primal(end, 2); % 获取最终的原始问题的目标函数值
    num_iters = size(iters_primal, 1); % 获取迭代的次数
    info = struct('iters', iters_primal, 'iters_dual', iters_dual); % 创建一个结构体来保存迭代信息
    out.fval = primal_obj_value; % 将最终的原始问题的目标函数值保存到输出结构体中
    out.ite_primal = iters_primal; % 将原始问题的迭代次数保存到输出结构体中
    out.ite_dual = iters_dual; % 将对偶问题的迭代次数保存到输出结构体中
    
    la=-la; % 返回对偶问题的解

end

% 投影函数
function v_out = project(v, mu)
    norm_v = vecnorm(v, 2, 2); % 计算每一行的范数
    norm_v(norm_v < mu) = mu;  % 将范数小于mu的值替换为mu
    v_out = v .* (mu ./ norm_v); % 返回v与mu/norm_v的按元素相乘的结果
end

% 原始问题的目标函数
function obj = primal_obj(A,b,mu,X)
    obj = 0.5 * norm(A * X - b, 'fro')^2 + mu * sum(vecnorm(X, 2, 2)); % 计算目标函数的值
end

% 对偶问题的目标函数
function obj = dual_obj(b,z)
    obj = 0.5 * norm(z, 'fro')^2 + sum(sum(z .* b)); % 计算目标函数的值
end
