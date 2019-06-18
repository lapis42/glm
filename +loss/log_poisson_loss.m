function [L, dL, H] = log_poisson_loss(w, X, y, dt, epsilon)

if nargin < 5
    epsilon = 0;
end

X_proj = X * w;
lambda = exp(X_proj) * dt;

% compute negative log-likelihood, 0.092-0.101 s
L = - y' * X_proj + sum(lambda);
if epsilon > 0
    L = L + epsilon * (w(2:end)' * w(2:end));
end

% compute gradient, 0.131-0.135 s
if nargout >= 2
    dL = X' * (lambda - y);
    if epsilon > 0
        dL = dL + [0; 2 * epsilon * w(2:end)];
    end
    
    % compute Hessian, 0.65-0.72 s...
    if nargout >= 3
        H = (lambda .* X)' * X;
        if epsilon > 0
            H = H + 2 * epsilon * diag([0;ones(length(w(2:end)), 1)]);
        end
    end
end
