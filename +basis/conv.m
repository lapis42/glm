function X = conv(x, filter, offset)
if nargin < 3
    offset = 0;
end

[~, n_type] = size(x);
[n_time, n_filter] = size(filter);

offset = min(offset);

if offset < 0
    x = [x; zeros(-offset, n_type)];
elseif offset > 0
    x = [zeros(offset, n_type); x];
end

n_x = size(x, 1);
X = zeros(n_x + n_time - 1, n_type * n_filter);
for i_type = 1:n_type
    X(:, (1:n_filter) + (i_type - 1) * n_filter) = conv2(x(:, i_type), filter);
end


if offset < 0
    X = X(-offset+1:end-n_time+1, :);
elseif offset > 0
    X = X(1:end-offset-n_time+1, :);
else
    X = X(1:end-n_time+1, :);
end