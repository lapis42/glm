function [X, prm] = build_design_matrix(X_cell, normalize)
%BUILD_DESIGN_MATRIX adds constant column and normalizes the design matrix.

if nargin < 2, normalize = false; end
assert(all(diff(cellfun(@(x) size(x, 1), X_cell)) == 0)); % make sure sample size is all the same.


X_cell = X_cell(:)';

prm = struct();
prm.n_sample = size(X_cell{1}, 1);
prm.n_type = length(X_cell);
prm.n_var = [1, cellfun(@(x) size(x, 2), X_cell)];
prm.n_var_sum = sum(prm.n_var);
cum_var = [0, cumsum(prm.n_var(1:end-1))];
prm.index = cellfun(@(x, y) (1:x)' + y, num2cell(prm.n_var), num2cell(cum_var), 'UniformOutput', false)';
prm.normalize = normalize;


X = [ones(prm.n_sample, 1), cell2mat(X_cell(:)')]; % add constant column


% Normalize design matrix (prm is needed to denormalize later)
if normalize
    prm.mean = mean(X, 1);
    prm.std = std(X, 1);
    prm.mean(1) = 0;
    prm.std(1) = 1;
    if any(prm.std < 1e-10)
        warning('std too small');
    end
    X = (X - prm.mean) ./ prm.std;
else
    prm.mean = zeros(1, prm.n_var_sum);
    prm.std = ones(1, prm.n_var_sum);
end
    
