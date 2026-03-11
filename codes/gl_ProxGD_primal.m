% 定义函数，输入包括初始解x0，数据矩阵A，观测向量b，正则化参数mu和选项参数opts
function [x,iter,out] = gl_ProxGD_primal(x0, A, b, mu, opts)
    obj_fun = @(z, tmu) 1/2 * norm(A*z-b, 'fro')^2 + tmu * sum(sqrt(sum(z.^2,2))); % 定义目标函数
    grad_fx = @(z) A.'* (A * z - b); % 定义梯度函数
    prox_hx = @(z,t) diag(max(0,1 - t * sqrt(sum(z.^2, 2)).^-1)) * z; % 定义近端函数
    iter = 0;
    x = x0;
    v = x0;
    scale = 2^16;
    scale_mu = scale * mu;
    tau = 1e-3;
    for i = 1:1000
        iter = iter + 1;
        y = x - tau * grad_fx(x); % 计算梯度步
        x = prox_hx(y, scale_mu * tau); % 近端操作
        if rem(iter,100)==0
            scale = max(scale / 4, 1);
            scale_mu = scale * mu;
            if scale <= 1
                break;
            end
        end
    end
    obj = obj_fun(x, mu);
    tau = 1e-4;
    x(abs(x)<1e-3) = 0;
    for i = 1:2000
        iter = iter + 1;
        y = x - tau * grad_fx(x); % 计算梯度步
        x = prox_hx(y, mu * tau); % 近端操作
        new_obj = obj_fun(x, mu);
        x(abs(x)<1e-5) = 0;
        if abs(new_obj-obj) < 1e-12
            break
        end
        obj = new_obj;
    end

    out.fval = 1/2 * norm(A*x-b, 'fro')^2 +  mu * sum(sqrt(sum(x.^2,2))); % 计算最终的目标函数值
end
