% 定义函数，输入包括初始解x0，数据矩阵A，观测向量b，正则化参数mu和选项参数opts
function [x,iter,out] = gl_mosek(x0, A, b, mu, opts)
    shape_x = size(x0); % 获得x0的大小
    shape_t = size(b); % 获得b的大小

    l = shape_x(2);
    len_x = shape_x(1) * shape_x(2);
    len_t = shape_t(1) * shape_t(2);
    len_z = shape_x(1);
    len_var = len_x + len_t + len_z;
    len_exvar = 2;
    [r, res] = mosekopt('symbcon');
    %% objective function
    prob.c = zeros(1, len_var + len_exvar);
    prob.c(len_x + len_t+1:len_var) = mu;
    prob.c(len_var + 1) = 0.5;

    prob.a = sparse(len_t + 1, len_var + len_exvar);
    for i1 = 1: shape_t(1)
        for i2 = 1: shape_t(2)
            shift_t = (i1 - 1) * shape_t(2) + i2;
            prob.a(shift_t, len_x + shift_t) = -1;
            for u = 1:shape_x(1)
                shift_x = u * shape_x(2) + i2;
                prob.a(shift_t, shift_x) = A(i1,u);
            end
        end
    end
    prob.a(len_t + 1, len_var + len_exvar) = 1;
    b_vec = b';
    b_vec = b_vec(:);
    prob.blc = [b_vec; 1];
    prob.buc = [b_vec; 1];
    %% conic constraints
    prob.cones.type   = [res.symbcon.MSK_CT_QUAD * ones(1,len_z), res.symbcon.MSK_CT_RQUAD];
    prob.cones.sub    = zeros(1, (shape_x(2) + 1) * len_z + len_t + 2);
    prob.cones.subptr = linspace(1, (shape_x(2) + 1) * (len_z) + 1,len_z+1);
    for i = 1:len_z
        pos = (i-1) * (shape_x(2) + 1) + 1;
        prob.cones.sub(pos) = len_x + len_t + i;
        prob.cones.sub(pos+1: pos+shape_x(2)) = linspace((i-1) * shape_x(2) + 1, i * shape_x(2), shape_x(2));
    end
    prob.cones.sub((shape_x(2) + 1) * len_z + 1) = len_var + 1;
    prob.cones.sub((shape_x(2) + 1) * len_z + 2) = len_var + 2;
    prob.cones.sub((shape_x(2) + 1) * len_z + 3: (shape_x(2) + 1) * len_z + len_t + 2) = linspace(len_x + 1,len_x + len_t,len_t);
    [r,res]=mosekopt('minimize',prob); % 调用Mosek优化器进行优化
    iter = -1;
    out.fval = res.sol.itr.pobjval; % 输出目标函数值
    x = res.sol.itr.xx(3:len_x + 2, 1);
    x = reshape(x, shape_x(2), shape_x(1));
    x = x'; % 输出最优解
end
