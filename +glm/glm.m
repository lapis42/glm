function out = glm(X_cell, y, varargin)
paramNames = {'order', 'lambda', 'lambda_range', 'n_lambda', 'remove', 'grid', 'parallel'};
paramDflts = {0, [], [], [], false, 'on', true};
[order, lambda, lambda_range, n_lambda, remove, grid_option, parallel] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});

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


% Run GLM with train and test dataset seperation
if n_lambda_grid > 1
    deviance_mean = zeros(1, n_lambda_grid);
    n_train = ceil(0.8 * prm.n_sample);
    n_test = prm.n_sample - n_train;
    train = randsample([true(n_train, 1); false(n_test, 1)], prm.n_sample);
    Xr = X(train, :); yr = y(train, :);
    Xt = X(~train, :); yt = y(~train, :);

    if (parallel)
        parfor i = 1:n_lambda_grid
            aLL = repelem([0, lambda_grid(i, :)], prm.n_var)' .* D;
            lfunc = @(w) loss.log_poisson_loss(w, Xr, yr, aLL);
            w1 = fminunc(lfunc, w0, opts);
            deviance_mean(i) = devi(yt, Xt * w1) / n_test;
        end
    else
        for i = 1:n_lambda_grid
            aLL = repelem([0, lambda_grid(i, :)], prm.n_var)' .* D;
            lfunc = @(w) loss.log_poisson_loss(w, Xr, yr, aLL);
            w1 = fminunc(lfunc, w0, opts);
            deviance_mean(i) = devi(yt, Xt * w1) / n_test;
        end
    end


    % Calculate best lambda
    if strcmp(grid_option, 'each')
        lambda_min = zeros(1, prm.n_type);
        deviance_each = mat2cell(deviance_mean, 1, lambda_size);
        for i = 1:prm.n_type
            lambda_min(i) = lambda{i}(find(deviance_each{i} <= min(deviance_each{i}), 1, 'last')); % pick the largest number
        end
    else
        lambda_min = lambda_grid(deviance_mean <= min(deviance_mean), :);
    end
    aLL = repelem([0, lambda_min], prm.n_var)' .* D;
else
    aLL = repelem([0, lambda_grid], prm.n_var)' .* D;
end


% Final fit using full dataset and selected lambda
lfunc = @(w) loss.log_poisson_loss(w, X, y, aLL);
w = fminunc(lfunc, w0, opts);
deviance = sum(devi(y, X * w));
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
if n_lambda_grid > 1
    out.lambda = lambda;
    out.lambda_min = lambda_min;
    if strcmp(grid_option, 'each')
        out.deviance_mean_test = deviance_each;
    else
        out.deviance_mean_test = reshape(deviance_mean, lambda_size);
    end
else
    out.lambda = lambda_grid;
end




function result = devi(yy, eta)
deveta = yy .* eta - exp(eta);
devy = yy .* log(yy + (yy==0)) - yy;
result = sum(2 * (devy - deveta));
