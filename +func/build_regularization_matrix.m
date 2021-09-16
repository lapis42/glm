function [D, prm] = build_regularization_matrix(order, prm)
%BUILD_REGULARIZATION_MATRIX builds Tikhonov regularization matrix.
%   Regularization order: 0-zeroth (=ridge regression), 1-first order (=smoothing prior)
prm.order = order;

% Build Tikhonov regularization matrix
Ls = cell(prm.n_type+1, 1);
Ls{1} = 0;
for ii = 2:prm.n_type+1
    if order == 0
        Ls{ii} = eye(prm.n_var(ii));
    else
        Ls{ii} = diff(eye(prm.n_var(ii)), order) / 2^order;
    end
end
L = blkdiag(Ls{:});
D = L' * L;
