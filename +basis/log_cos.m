function [bump, bin, cos_func] = log_cos(n_bump, range, dt, bias, normalize)

if nargin < 2
    range = [0, 100];
end
if length(range) == 1
    range = [0, range];
end
assert(range(1) >=0, 'range should be larger than 0');

if nargin < 3
    dt = 1;
end

bias_min = 0.1;
if nargin < 4 || bias < bias_min
    bias = bias_min;
end
bias = bias * diff(range) / 100;

if nargin < 5
    normalize = true;
end

log_func = @(x) log(x + bias);
exp_func = @(x) exp(x) - bias;

range_log = log_func(range);
gap_log = diff(range_log) / (n_bump + 1);

bin = (range(1):dt:range(2))';
peak_log = range_log(1):gap_log:range_log(2)-2*gap_log;

cos_func = @(x) raised_cosine(x, peak_log, gap_log, log_func);
bump = cos_func(bin);

if normalize
    bump = bump ./ sum(bump, 1);
end


function bump = raised_cosine(x, peak, gap_log, log_func)
x = x(:);
n_x = length(x);

peak = peak(:)';
n_peak = length(peak);

x_mat = repmat(x, 1, n_peak);
peak_mat = repmat(peak, n_x, 1);

cos_func = @(x, peak) (cos(max(-pi, min(pi, (log_func(x) - peak) * pi / (2 * gap_log)))) + 1) / 2;

bump = cos_func(x_mat, peak_mat);