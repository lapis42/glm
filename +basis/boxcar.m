function [boxcar_func, bin, bump, peak] = boxcar(n_bump, peak_range, bin_size, cut_edge, normalize)
if nargin < 1, n_bump = 11; end
if nargin < 2, peak_range = [0, 100]; end
if numel(peak_range) == 1, peak_range = [0, peak_range]; end
if nargin < 3, bin_size = 1; end
if nargin < 4, cut_edge = false; end
if nargin < 5, normalize = false; end


gap = diff(peak_range) / (n_bump - 1);
peak = peak_range(1):gap:peak_range(2);
boxcar_func = @(x) x >= peak - gap/2 & x < peak + gap/2;

if nargout > 1
    if cut_edge
        bin = (peak_range(1):bin_size:peak_range(2))';
    else
        bin = (peak_range(1)-gap/2:bin_size:peak_range(2)+gap/2)';
    end
end

if nargout > 2
    bump = boxcar_func(bin);
    if normalize
        bump = bump ./ sum(bump, 1);
    end
end
