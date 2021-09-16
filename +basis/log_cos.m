function [cos_func, bin, bump, peak] = log_cos(n_bump, range, bin_size, bias, normalize)
if nargin < 1, n_bump = 10; end
if nargin < 2, range = [0, 100]; end
if length(range) == 1, range = [0, range]; end
assert(range(1) >=0, 'range should be larger than 0');
if nargin < 3, bin_size = 1; end

bias_min = 0.1;
if nargin < 4 || bias < bias_min
    bias = bias_min;
end
bias = bias * diff(range) / 100;

if nargin < 5, normalize = false; end


range_log = log(range + bias);
gap_log = diff(range_log) / (n_bump + 1);
peak_log = range_log(1):gap_log:range_log(2)-2*gap_log;
peak = exp(peak_log) - bias;
cos_func = @(x) (cos(max(-pi, min(pi, (log(x + bias) - peak_log) * pi / (2 * gap_log)))) + 1) / 2;

if nargout > 1
    bin = (range(1):bin_size:range(2))';
end

if nargout > 2
    bump = cos_func(bin);
    if normalize
        bump = bump ./ sum(bump, 1);
    end
end
