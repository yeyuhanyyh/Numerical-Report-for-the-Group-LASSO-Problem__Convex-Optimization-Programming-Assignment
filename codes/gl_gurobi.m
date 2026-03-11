% 定义函数，输入包括初始解x0，数据矩阵A，观测向量b，正则化参数mu和选项参数opts
function [x,iter,out] = gl_gurobi(x0, A, b, mu, opts)
    shape_x = size(x0);
    shape_t = size(b);

    l = shape_x(2);
    len_x = shape_x(1) * shape_x(2);
    len_t = shape_t(1) * shape_t(2);
    len_z = shape_x(1);
    len_var = len_x + len_t + len_z;
    len_exvar = 2;
    %% objective function
    model.obj = zeros(1, len_var + len_exvar);
    model.obj(len_x + len_t+1:len_var) = mu;
    model.obj(len_var + 1) = 0.5;
    %% lower bound
    model.lb = [-inf(len_x + len_t, 1);zeros(len_z+2,1)];
    %% linear constraints
    model.A = sparse(len_t + 1, len_var + len_exvar);
    for i1 = 1: shape_t(1)
        for i2 = 1: shape_t(2)
            shift_t = (i1 - 1) * shape_t(2) + i2;
            model.A(shift_t, len_x + shift_t) = -1;
            for u = 1:shape_x(1)
                shift_x = u * shape_x(2) + i2;
                model.A(shift_t, shift_x) = A(i1,u);
            end
        end
    end
    model.A(len_t + 1, len_var + len_exvar) = 1;
    b_vec = b';
    b_vec = b_vec(:);
    model.rhs = [b_vec; 1];
    model.sense = '=';
    %% quadratic constraints
    for i = 1: len_z
        model.quadcon(i).Qc = sparse(len_var + len_exvar,len_var + len_exvar);
        start_pos = (i-1) * shape_x(2) + 1;
        end_pos = i * shape_x(2);
        model.quadcon(i).Qc(start_pos:end_pos, start_pos:end_pos) = eye(shape_x(2));
        shift_z = len_x + len_t + i;
        model.quadcon(i).Qc(shift_z, shift_z) = -1;
        model.quadcon(i).q  = sparse(len_var + len_exvar,1);
        model.quadcon(i).rhs = 0.0;
    end
    model.quadcon(len_z + 1).Qc = sparse(len_var + len_exvar,len_var + len_exvar);
    model.quadcon(len_z + 1).Qc(len_x + 1:len_x + len_t, len_x + 1:len_x + len_t) = eye(len_t);
    model.quadcon(len_z + 1).Qc(len_var + 1, len_var + 2) = -1;
    model.quadcon(len_z + 1).q  = sparse(len_var + len_exvar,1);
    model.quadcon(len_z + 1).rhs = 0.0;
    result = gurobi(model); % 调用Gurobi优化器求解
    out.fval = result.objval;
    x = result.x(3: shape_x(1) * shape_x(2)+2, 1);
    x = reshape(x, shape_x(2), shape_x(1));
    x = x';
    iter = -1;
end
