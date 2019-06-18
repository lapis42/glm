function [bump, bin, cos_func] = linear_cos(n_bump, range, dt, normalize)
if nargin < 1
    n_bump = 10;
end

if nargin < 2
    range = [-1000, 1000];
end
if length(range) == 1
    range = [-range, range];
end

if nargin < 3
    dt = 1;
end

if nargin < 4
    normalize = true;
end

gap = diff(range) / (n_bump + 3);

bin = (range(1):dt:range(2))';
peak = range(1)+2*gap:gap:range(2)-2*gap;

cos_func = @(x) raised_cosine(x, peak, gap);
bump = cos_func(bin);

if normalize
    bump = bump ./ sum(bump, 1);
end


function bump = raised_cosine(x, peak, gap)
x = x(:);
n_x = length(x);

peak = peak(:)';
n_peak = length(peak);

x_mat = repmat(x, 1, n_peak);
peak_mat = repmat(peak, n_x, 1);

cos_func = @(x, peak) (cos(max(-pi, min(pi, (x - peak) * pi / (2 * gap)))) + 1) / 2;

bump = cos_func(x_mat, peak_mat);