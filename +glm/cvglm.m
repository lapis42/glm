function out = cvglm(X_cell, y, varargin)
paramNames = {'order', 'lambda', 'lambda_range', 'n_lambda', 'cv', 'remove', 'grid', 'parallel'};
paramDflts = {0, [], [], [], 5, false, 'on', true};
[order, lambda, lambda_range, n_lambda, k_fold, remove, grid_option, parallel] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});

if isempty(lambda_range)
    lambda_range = [1e-4, 1e2] * 100^order;
end

if isempty(n_lambda)
    if strcmp(grid_option, 'on')
        n_lambda = 13;
    else
        n_lambda = 100;
    end
end


% Build design matrix and regularization matrix
[X, prm] = func.build_design_matrix(X_cell);
[D, prm] = func.build_regularization_matrix(order, prm);


% Remove sample without any coding (caution!)
if (remove)
    out_sample = all(X(:, 2:end) == 0, 2);
    X(out_sample, :) = [];
    y(out_sample, :) = [];
end
prm.n_sample = length(y);
prm.n_spike = sum(y);
prm.grid_option = grid_option;


% Build lambdas
[lambda_grid, lambda, lambda_size] = func.build_lambda_grid(lambda_range, n_lambda, lambda, prm.n_type, grid_option);
n_lambda_grid = size(lambda_grid, 1);


% Initialize weights
warning off
w0 = X \ y;
warning on
if max(w0) > 1e4
    w0 = ones(prm.n_var_sum, 1); % if w0 is unstable, reset to 1.
end


% Optimizer options
opts = optimset('algorithm','trust-region', ...
    'Gradobj','on', ...
    'Hessian','on', ...
    'display', 'notify', ...
    'maxiter', 100);


% Select K-fold subsets
population = cat(2, repmat(1:k_fold, 1, floor(prm.n_sample/k_fold)), 1:mod(prm.n_sample, k_fold));
foldid = population(randperm(length(population), prm.n_sample));


% Optimization
cv_deviance = zeros(k_fold, n_lambda_grid);
if (parallel)
    parfor ii = 1:k_fold*n_lambda_grid
        i = mod(ii-1, k_fold) + 1;
        j = ceil(ii/k_fold);

        which = foldid == i;
        Xr = X(~which, :); yr = y(~which, :);
        Xt = X(which, :); yt = y(which, :);

        aLL = repelem([0, lambda_grid(j, :)], prm.n_var)' .* D; % = alpha * L' * L
        lfunc = @(w) loss.log_poisson_loss(w, Xr, yr, aLL);
        w1 = fminunc(lfunc, w0, opts);
        cv_deviance(ii) = devi(yt, Xt * w1);
    end 
else
    for ii = 1:k_fold*n_lambda_grid
        i = mod(ii-1, k_fold) + 1;
        j = ceil(ii/k_fold);

        which = foldid == i;
        Xr = X(~which, :); yr = y(~which, :);
        Xt = X(which, :); yt = y(which, :);

        aLL = repelem([0, lambda_grid(j, :)], prm.n_var)' .* D; % = alpha * L' * L
        lfunc = @(w) loss.log_poisson_loss(w, Xr, yr, aLL);
        w1 = fminunc(lfunc, w0, opts);
        cv_deviance(ii) = devi(yt, Xt * w1);
    end 
end


% Calculate best lambda
cv_deviance_mean = sum(cv_deviance, 1) / prm.n_sample;
if strcmp(grid_option, 'each')
    lambda_min = zeros(1, prm.n_type);
    cv_deviance_each = mat2cell(cv_deviance_mean, 1, lambda_size);
    for i = 1:prm.n_type
        lambda_min(i) = lambda{i}(cv_deviance_each{i} <= min(cv_deviance_each{i}));
    end
else
    lambda_min = lambda_grid(cv_deviance_mean <= min(cv_deviance_mean), :);
end
    


% Final fit using full dataset and selected lambda
aLL = repelem([0, lambda_min], prm.n_var)' .* D;
lfunc = @(w) loss.log_poisson_loss(w, X, y, aLL);
w = fminunc(lfunc, w0, opts);
deviance = devi(y, X * w);
deviance0 = devi(y, log(mean(y)));


% Seperate weights
ws = cell(prm.n_type, 1);
for i = 1:prm.n_type
    ws{i} = w(prm.index{i + 1});
end

out = struct();
out.prm = prm;
out.w0 = w(1);
out.w = ws;
out.deviance0 = deviance0;
out.deviance = deviance;
out.deviance_mean = deviance / prm.n_sample;
out.deviance0_mean = deviance0 / prm.n_sample;
out.deviance_spike = deviance / prm.n_spike;
out.deviance0_spike = deviance0 / prm.n_spike;
out.p = 1 - chi2cdf(deviance0 - deviance, sum(w ~= 0, 1) - 1);
out.r2 = 1 - deviance / deviance0;
out.lambda = lambda;
out.lambda_min = lambda_min;
if strcmp(grid_option, 'each')
    out.cv_deviance_mean = cv_deviance_each;
else
    out.cv_deviance_mean = reshape(cv_deviance_mean, lambda_size);
end




function result = devi(yy, eta)
deveta = yy .* eta - exp(eta);
devy = yy .* log(yy + (yy==0)) - yy;
result = sum(2 * (devy - deveta));
