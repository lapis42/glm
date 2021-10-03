function [lambda_grid, lambda, lambda_size] = build_lambda_grid(lambda_range, n_lambda, lambda, n_type, grid_option)
if nargin < 1, lambda_range = [1e-4, 1e2]; end
if nargin < 2, n_lambda = 100; end
if nargin < 3, lambda = []; end
if nargin < 4, n_type = 1; end
% grid_option
%    off: use the same lambdas for all variables
%    each: search lambda for each variable while using fixed(=zero) lambda for other variable. consider this if you have too many variables.
%    on: do grid search for all variables
if nargin < 5, grid_option = 'on'; end


% check whether lambda_range are given separately for each variable
if iscell(lambda_range)
    assert(length(lambda_range) == n_type);
else
    lambda_range = repmat({lambda_range}, 1, n_type);
end


% check whether n_lambda is given separately for each variable
if numel(n_lambda) == 1
    n_lambda = repmat(n_lambda, 1, n_type);
else
    assert(length(n_lambda) == n_type);
end


% if lambda is given separately, use that.
if iscell(lambda)
    assert(length(lambda) == n_type);
elseif isempty(lambda)
    lambda = cell(1, n_type);
    for i = 1:n_type
        lambda{i} = logspace(log10(lambda_range{i}(1)), log10(lambda_range{i}(2)), n_lambda(i));
    end
else
    % if lambda is given, just overwrite...
    lambda = repmat({lambda}, 1, n_type);
end



if (strcmp(grid_option, 'on') && n_type > 1)
    [lambda_grid, lambda_size] = func.ndgrid(lambda, n_type);
elseif (strcmp(grid_option, 'each'))
    lambda_grid = [];
    for i = 1:n_type
        x = zeros(size(lambda{i}, 2), n_type);
        x(:, i) = lambda{i};
        lambda_grid = [lambda_grid; x];
    end
    lambda_size = cellfun(@length, lambda);
else
    lambda_grid = repmat(lambda{1}', 1, n_type);
    lambda_size = size(lambda{1});
end
