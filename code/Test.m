% 设置随机数生成器的种子以保证结果的可重复性
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);

% 生成测试数据
n = 512; % 数据的维度
m = 256; % 数据的数量
A = randn(m,n); % 生成随机数据矩阵A
k = round(n*0.1); l = 2; 
p = randperm(n); p = p(1:k);
u = zeros(n,l); u(p,:) = randn(k,l); 
b = A*u; 
mu = 1e-2; % 正则化参数
x0 = randn(n, l); % 初始解

% 定义评价函数
errfun_exact = @(x) norm(x - u, 'fro') / (1 + norm(u,'fro')); % 计算解的误差
sparisity = @(x) sum(abs(x(:)) > 1E-6 * max(abs(x(:)))) /(n*l); % 计算解的稀疏性
opts = []; % 选项参数

% 运行优化算法
tic; % 开始计时
%第2问
%[x, iter, out] = gl_gurobi(x0, A, b, mu, opts);
%[x, iter, out] = gl_mosek(x0, A, b, mu, opts);

%第1问
%[x, iter, out] = gl_cvx_mosek(x0, A, b, mu, opts);
%[x, iter, out] = gl_cvx_gurobi(x0, A, b, mu, opts);

%第3问
%[x, iter, out] = gl_SGD_primal(x0, A, b, mu, opts); %a
%[x, iter, out] = gl_GD_primal(x0, A, b, mu, opts); %b
%[x, iter, out] = gl_ProxGD_primal(x0, A, b, mu, opts); %d
%[x, iter, out] = gl_FProxGD_primal(x0, A, b, mu, opts); %e
%[x, iter, out] = gl_ALM_dual(x0, A, b, mu, opts); %f
%[x, iter, out] = gl_ADMM_dual(x0, A, b, mu, opts);%g
[x, iter, out] = gl_ADMM_primal(x0, A, b, mu, opts);%h
t = toc;
% 输出结果
fprintf('Result## cputime: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E\n', t, iter, out.fval, sparisity(x), errfun_exact(x));
