% 定义函数，输入包括初始解x0，数据矩阵A，观测向量b，正则化参数mu和选项参数opts
function [x,iter,out] = gl_SGD_primal(x0, A, b, mu, opts)
    x = x0;
    e = 1e-6;
    alpha = 1e-3;
    iter = 0;
    record_obj = [];
    obj = 1/2 * norm(A*x-b, 'fro')^2 + mu * sum(sqrt(sum(x.^2,2)));
    for n = 8:-1:1
        mu_1 = 2^n * mu;
        for i = 1:1000
            sg = subgrad(x, A, b, mu_1, e); % 计算次梯度
            x = x - alpha * sg; % 更新解
            x(abs(x)<1e-3) = 0;
            new_obj = 1/2 * norm(A*x-b, 'fro')^2 + mu * sum(sqrt(sum(x.^2,2)));
            record_obj = [record_obj new_obj];
            if(abs(new_obj-obj) < 1e-13)
                break;
            end
            obj = new_obj;
        end
        iter = iter + i;
        obj = 1/2 * norm(A*x-b, 'fro')^2 + mu * sum(sqrt(sum(x.^2,2)));
    end
    alpha = 1e-5;
    obj = 1/2 * norm(A*x-b, 'fro')^2 + mu * sum(sqrt(sum(x.^2,2)));
    for i = 1:10000
        sg = subgrad(x, A, b, mu, e); % 计算次梯度
        x = x - alpha * sg; % 更新解
        x(abs(x)<1e-5) = 0;
        new_obj = 1/2 * norm(A*x-b, 'fro')^2 + mu * sum(sqrt(sum(x.^2,2)));
        record_obj = [record_obj new_obj];
        if(abs(new_obj-obj) < 1e-13)
            break;
        end
        obj = new_obj;
    end
    iter = iter + i;

    obj = 1/2 * norm(A*x-b, 'fro')^2 + mu * sum(sqrt(sum(x.^2,2)));

    out.fval = obj;
    out.status = 0;
    out.record_obj = record_obj;
end
