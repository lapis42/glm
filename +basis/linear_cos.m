function [cos_func, bin, bump, peak] = linear_cos(n_bump, range, bin_size, cut_edge, normalize)
if nargin < 1, n_bump = 10; end
if nargin < 2, range = [-1000, 1000]; end
if length(range) == 1, range = [0, range]; end
if nargin < 3, bin_size = 1; end
if nargin < 4, cut_edge = false; end
if nargin < 5, normalize = false; end


gap = diff(range) / (n_bump - 1);
peak = range(1):gap:range(2);
cos_func = @(x) (cos(max(-pi, min(pi, (x - peak) * pi / (2 * gap)))) + 1) / 2;

if nargout > 1
    if cut_edge
        bin = (range(1):bin_size:range(2))';
    else
        bin = (range(1)-2*gap:bin_size:range(2)+2*gap)';
    end
end

if nargout > 2
    bump = cos_func(bin);
    if normalize
        bump = bump ./ sum(bump, 1);
    end
end
