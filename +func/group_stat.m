function [y_mean, y_sem, x_bin] = group_stat(x, y, x_edge)
[x_count, ~, idx] = histcounts(x, x_edge);
out_idx = idx==0;
x(out_idx) = [];
y(out_idx) = [];
idx(out_idx) = [];

x_step = x_edge(2) - x_edge(1);
x_bin = x_edge(1:end-1) + x_step / 2;
n_bin = length(x_bin);

y_mean = accumarray(idx, y, [n_bin, 1], @mean);
y_sem = accumarray(idx, y, [n_bin, 1], @(x) std(x) / sqrt(length(x)));

out_count = (x_count' == 0 | y_mean == 0);
x_bin(out_count) = [];
y_mean(out_count) = [];
y_sem(out_count) = [];

