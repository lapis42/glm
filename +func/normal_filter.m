function [X_filtered, f, t] = normal_filter(X, sigma, bin_size, a)
narginchk(3, 4);
assert(sigma > bin_size);
if nargin < 4 || isempty(a)
    a = 4;
end

% make gaussian filter
s = sigma / bin_size;
L = 2 * a * s + 1;
t = (0:(L-1))' - (L-1)/2;
f = exp(-0.5*(a*t/(L/2)).^2);
f = f / sum(f);

% convolution
X_filtered = conv2(X, f, 'same');