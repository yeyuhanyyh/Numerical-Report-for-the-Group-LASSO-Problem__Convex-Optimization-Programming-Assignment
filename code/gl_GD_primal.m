% 定义函数，输入包括初始解x0，数据矩阵A，观测向量b，正则化参数mu和选项参数opts
function [x,iter,out] = gl_GD_primal(x0, A, b, mu, opts)
    x = x0; % 初始解
    sm_alpha = 1e-2;
    alpha = 1e-3;
    iter = 0;
    e = 1e-6;
    record_obj = [];
    obj = 1/2 * norm(A*x-b, 'fro')^2 + mu * sum(sqrt(sum(x.^2,2))); % 计算目标函数值
    for n = 8:-1:1
        mu_1 = 2^n * mu;
        sm_alpha = sm_alpha - 1e-3;
        for i = 1:1000
            sg = sm_grad(x, A, b, mu_1, sm_alpha); % 计算梯度
            x = x - alpha * sg; % 更新x
            x(abs(x)<1e-3) = 0;
            new_obj = 1/2 * norm(A*x-b, 'fro')^2 + mu * sum(sqrt(sum(x.^2,2))); % 计算新的目标函数值
            record_obj = [record_obj new_obj];
            if(abs(new_obj-obj) < 1e-13)
                break;
            end
            obj = new_obj;
        end
        iter = iter + i;
        obj = 1/2 * norm(A*x-b, 'fro')^2 + mu * sum(sqrt(sum(x.^2,2)));
    end
    sm_alpha = 1e-6;
    alpha = 1e-5;
    obj = 1/2 * norm(A*x-b, 'fro')^2 + mu * sum(sqrt(sum(x.^2,2)));
    for i = 1:10000
        sg = subgrad(x, A, b, mu, e); % 计算梯度
        x = x - alpha * sg; % 更新x
        x(abs(x)<1e-5) = 0;
        new_obj = 1/2 * norm(A*x-b, 'fro')^2 + mu * sum(sqrt(sum(x.^2,2))); % 计算新的目标函数值
        record_obj = [record_obj new_obj];
        if(abs(new_obj-obj) < 1e-13)
            break;
        end
        obj = new_obj;
    end
    iter = iter + i;
    obj = 1/2 * norm(A*x-b, 'fro')^2 + mu * sum(sqrt(sum(x.^2,2))); % 计算最终的目标函数值

    out.fval = obj;
    out.status = 0;
    out.record_obj = record_obj; % 记录每次迭代的目标函数值
end


