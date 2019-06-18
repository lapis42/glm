function [y_mean, y_sem] = group_stat2(y, group)
% y: n_time_bin x n_trial
% group: n_trial

assert(size(y, 2) == numel(group));

out_group = group == 0;
y(:, out_group) = [];
group(out_group) = [];


n_bin = size(y, 1);
n_group = max(group);

[y_mean, y_sem] = deal(NaN(n_bin, n_group));
for i_group = 1:n_group
    in_group = group == i_group;
    n_in = double(sum(in_group));
    if n_in == 0; continue; end
    
    y_mean(:, i_group) = mean(y(:, in_group), 2);
    y_sem(:, i_group) = std(y(:, in_group), 0, 2) / sqrt(n_in);
end
