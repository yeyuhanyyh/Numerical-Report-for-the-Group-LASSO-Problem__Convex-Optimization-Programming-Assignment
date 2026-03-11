% ADMM求解器，求解对偶问题
function [la, num_iters, out] = gl_ADMM_dual(x0, A, b, mu, opts)
    % 获取A的大小
    [m, n] = size(A);
    l = size(b, 2);

    % 初始化迭代列表
    iters_primal = [];
    iters_dual = [];
    % 初始化la
    la = -x0;

    % 设置参数t
    t = 0.1; 
    % 计算逆矩阵
    inv1 = inv(eye(m) + t * A * A');
    % 初始化v
    v = zeros(n, l);
    % 初始化缩放因子和缩放后的mu
    scale = 2^16; 
    scale_mu = scale * mu; 

    % 开始迭代
    for it = 1:10000
        % 每5次迭代，更新缩放因子和缩放后的mu
        if rem(it,5)==0 
            scale = max(scale / 4, 1); 
            scale_mu = scale * mu; 
        end
        % 更新z
        z = inv1 * (t * A * v - A * la - b);
        % 记录旧的v
        v0 = v;
        % 更新v
        v = project(A' * z + la / t, scale_mu);
        % 记录旧的la
        la0 = la;
        % 更新la
        la = la + t * (A' * z - v);

        % 记录迭代次数和目标函数值
        iters_primal = [iters_primal; it, primal_obj(A,b,scale_mu,-la)];
        iters_dual = [iters_dual; it, dual_obj(b,z)];

        % 计算对偶饱和度和对偶的对偶饱和度
        dual_sat = norm(A' * z - v, 'fro');
        dual_dual_sat = norm(A * (v - v0), 'fro');

        % 检查是否收敛
        if dual_sat < 1e-3 && dual_dual_sat < 1e-3
            break;
        end
    end

    % 返回迭代次数
    num_iters = size(iters_primal, 1);
    % 返回迭代信息
    out.iters = iters_primal;
    out.iters_dual = iters_dual;
    % 返回目标函数值
    out.fval = -dual_obj(b,z);
    % 更新la
    la = -la;

    % 在主函数中添加以下代码

end

% 投影函数，用于将向量投影到一个满足某种范数约束的集合上
function v_out = project(v, mu)
    norm_v = vecnorm(v, 2, 2); 
    norm_v(norm_v < mu) = mu;  
    v_out = v .* (mu ./ norm_v); 
end

% 原始问题的目标函数
function obj = primal_obj(A,b,mu,X)
    obj = 0.5 * norm(A * X - b, 'fro')^2 + mu * sum(vecnorm(X, 2, 2));
end

% 对偶问题的目标函数
function obj = dual_obj(b,z)
    obj = 0.5 * norm(z, 'fro')^2 + sum(sum(z .* b));
end

