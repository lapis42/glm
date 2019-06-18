function [X, prm] = normalize_add_constant(X_cell, normalize)
if nargin < 2
    normalize = 1;
end

n_var = [1, cellfun(@(x) size(x, 2), X_cell)];
cum_var = [0, cumsum(n_var(1:end-1))];

prm.index = cellfun(@(x, y) (1:x)' + y, num2cell(n_var), num2cell(cum_var), 'UniformOutput', false)';


X = cell2mat(X_cell);
prm.n_var = size(X, 2);

if normalize>0
    prm.mean = mean(X, 1);
    prm.std = std(X, 1);

    if any(prm.std < 1e-10)
        warning('std too small');
    end
else
    prm.mean = zeros(1, prm.n_var);
    prm.std = ones(1, prm.n_var);
end
    
X = [ones(size(X, 1), 1), (X - prm.mean) ./ prm.std];