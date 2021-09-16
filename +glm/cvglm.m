function out = cvglm(X_cell, y, varargin)
paramNames = {'order', 'cv', 'lambda_range', 'n_lambda'};
paramDflts = {0, 5, [1e-2, 1e4], 10};
[order, n_fold, lambda_range, n_lambda] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});


[X, prm] = func.build_design_matrix(X_cell);
[D, prm] = func.build_regularization_matrix(order, prm);


w0 = X \ y;
if max(w0) > 1e4
    % if w0 is unstable, reset to 1.
    w0 = ones(prm.n_var_sum, 1);
end


lambdas = logspace(log10(lambda_range(1)), log10(lambda_range(2)), n_lambda);
lambda_grid = func.ndgrid(lambdas, prm.n_type);
n_lambda_grid = size(lambda_grid, 1);


opts = optimset('algorithm','trust-region', ...
    'Gradobj','on', ...
    'Hessian','on', ...
    'display', 'notify', ...
    'maxiter', 100);


cv_deviance = zeros(n_fold, n_lambda_grid);
population = cat(2, repmat(1:n_fold, 1, floor(prm.n_sample/n_fold)), 1:mod(prm.n_sample, n_fold));
foldid = population(randperm(length(population), prm.n_sample));
n_foldid = histc(foldid, 1:n_fold);

parfor i = 1:n_fold
    which = foldid == i;
    Xr = X(~which, :); yr = y(~which, :);
    Xt = X(which, :); yt = y(which, :);

    for j = 1:n_lambda_grid
        aLL = repelem([0, lambda_grid(j, :)], prm.n_var)' .* D; % = alpha * L' * L
        lfunc = @(w) loss.log_poisson_loss(w, Xr, yr, aLL);
        w1 = fminunc(lfunc, w0, opts);
        cv_deviance(i, j) = mean(devi(yt, Xt * w1));
    end
end 

cv_deviance_mean = sum(n_foldid' .* cv_deviance, 1) / sum(n_foldid);
lambda_min = lambda_grid(cv_deviance_mean <= min(cv_deviance_mean), :);


% final fit using full dataset and selected lambda
aLL = repelem([0, lambda_min], prm.n_var)' .* D;
lfunc = @(w) loss.log_poisson_loss(w, X, y, aLL);
w = fminunc(lfunc, w0, opts);
deviance = sum(devi(y, X * w));


out = struct();
out.lambdas = lambdas;
out.lambda_grid = lambda_grid;
out.lambda_min = lambda_min;
out.w = w;
out.deviance = deviance;
out.cv_deviance_mean = cv_deviance_mean;
out.prm = prm;




function result = devi(yy, eta)
deveta = yy .* eta - exp(eta);
devy = yy .* log(yy + (yy==0)) - yy;
result = 2 * (devy - deveta);
