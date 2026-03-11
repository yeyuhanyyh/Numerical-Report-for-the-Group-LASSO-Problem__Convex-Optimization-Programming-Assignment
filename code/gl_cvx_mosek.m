% 定义函数，输入包括初始解x0，数据矩阵A，观测向量b，正则化参数mu和选项参数opts
function [x,iter,out] = gl_cvx_mosek(x0, A, b, mu, opts)
    shape = size(x0); % 获取x0的维度
    cvx_solver mosek; % 设置CVX的求解器为Mosek

    cvx_begin quiet % 开始CVX模型，quiet选项表示在求解过程中不显示输出
        variable x(shape(1), shape(2)) % 定义优化变量x
        fro_norm = norm(A * x - b, 'fro'); % 计算Frobenius范数
        obj = 1/2 * square_pos(fro_norm); % 定义优化目标的一部分
        for i = 1: shape(1) % 对于每一行
            obj = obj + mu * norm(x(i,:),2); % 加上正则化项
        end
        minimize(obj)  % 最小化目标函数
    cvx_end % 结束CVX模型

    iter = -1; % 在CVX中，我们无法获得迭代次数，所以这里设为-1
    out.fval = cvx_optval; % 输出最优值
    out.status = cvx_status; % 输出求解状态
end
