function [L, dL, ddL] = log_poisson_loss(w, X, y, D)
% L = -log p = sum(lambda) - y'Xw + 1/2*alpha*w'L'Lw
% dL = X' * (lambda - y) + alpha*L'L*w
% ddL = X' * diag(lambda) * X + alpha*L'L
% D = alpha*L'L


% compute negative log-likelihood
X_proj = X*w;
lambda = exp(X_proj);
L = sum(lambda) - y'*X_proj;
if nargin > 3
    L = L + w'*D*w/2;
end

% compute gradient
if nargout >= 2
    dL = X' * (lambda-y);
    if nargin > 3
        dL = dL + D*w;
    end
end
    
% compute Hessian
if nargout >= 3
    ddL = (lambda .* X)' * X;
    if nargin > 3
        ddL = ddL + D;
    end
end
