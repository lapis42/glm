n = 8;
bin_size = 150;
% b = 10;



% range = [0, bin_size*(1 - 1.5/n)];
range = [1, 100];
b = 20;

log_func = @(x) log(x + b);
exp_func = @(x) exp(x) - b;


range_log = log_func(range);
gap_log = diff(range_log) / (n - 1);


cos_func = @(x, peak) (cos(max(-pi, min(pi, (x - peak) * pi / (2 * gap_log)))) + 1) / 2;


bin_log = repmat(log_func((0:bin_size-1)'), 1, n);
peak_log = repmat(range_log(1):gap_log:range_log(2), bin_size, 1);

bump = cos_func(bin_log, peak_log);
bump = bump ./ sqrt(sum(bump.^2));

% subplot(2, 1, 1);
plot(bump);

% subplot(2, 1, 2);
% plot(flipud(gg0.ihbas));
