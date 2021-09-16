function out = glm(X_cell, y, varargin)
paramNames = {'order', 'lambda', 'lambda_range', 'n_lambda'};
paramDflts = {0, [], [1e-4, 1e2], 10};
[order, lambda, lambda_range, n_lambda] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});


[X, prm] = func.build_design_matrix(X_cell);
[D, prm] = func.build_regularization_matrix(order, prm);


w0 = X \ y;
if max(w0) > 1e4
    % if w0 is unstable, reset to 1.
    w0 = ones(prm.n_var_sum, 1);
end

% if lambda is given, just use that value
if ~isempty(lambda)
    if numel(lambda) == 1
        lambda = repmat(lambda, 1, prm.n_type);
    else
        lambda = lambda(:)';
    end
    assert(numel(lambda) == prm.n_type);
    n_lambda = 1;
end

if n_lambda > 1
    lambdas = logspace(log10(lambda_range(1)), log10(lambda_range(2)), n_lambda);
    lambda_grid = func.ndgrid(lambdas, prm.n_type);
    n_lambda_grid = size(lambda_grid, 1);
end


opts = optimset('algorithm','trust-region', ...
    'Gradobj','on', ...
    'Hessian','on', ...
    'display', 'notify', ...
    'maxiter', 100);


if n_lambda > 1
    deviance_mean = zeros(1, n_lambda_grid);
    n_train = ceil(0.8 * prm.n_sample);
    n_test = prm.n_sample - n_train;
    train = randsample([true(n_train, 1); false(n_test, 1)], prm.n_sample);
    Xr = X(train, :); yr = y(train, :);
    Xt = X(~train, :); yt = y(~train, :);

    for i = 1:n_lambda_grid
        aLL = repelem([0, lambda_grid(i, :)], prm.n_var)' .* D;
        lfunc = @(w) loss.log_poisson_loss(w, Xr, yr, aLL);
        w1 = fminunc(lfunc, w0, opts);
        deviance_mean(i) = mean(devi(yt, Xt * w1));
    end
    lambda_min = lambda_grid(deviance_mean <= min(deviance_mean), :);

    % final fit using full dataset and selected lambda
    aLL = repelem([0, lambda_min], prm.n_var)' .* D;
    lfunc = @(w) loss.log_poisson_loss(w, X, y, aLL);
    w = fminunc(lfunc, w0, opts);
    deviance = sum(devi(y, X * w));
else
    aLL = repelem([0, lambda], prm.n_var)' .* D;
    lfunc = @(w) loss.log_poisson_loss(w, X, y, aLL);
    w = fminunc(lfunc, w0, opts);
    deviance = sum(devi(y, X * w));
end


out = struct();
if n_lambda > 1
    out.lambdas = lambdas;
    out.lambda_grid = lambda_grid;
    out.lambda_min = lambda_min;
    out.cv_deviance_mean = cv_deviance_mean;
else
    out.lambda = lambda;
    out.deviance_mean = deviance_mean;
end
out.w = w;
out.deviance = deviance;
out.prm = prm;




function result = devi(yy, eta)
deveta = yy .* eta - exp(eta);
devy = yy .* log(yy + (yy==0)) - yy;
result = 2 * (devy - deveta);
