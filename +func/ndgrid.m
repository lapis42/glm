function [out, siz] = ndgrid(array,nd)
%NDGRID Rectangular grid in N-D space
%   OUT = NDGRID(ARRAY,ND) replicates the array and generates n-dimensional
%   grid.

if iscell(array)
    assert(length(array) == nd);
else
    array = repmat({array}, 1, nd);
end

siz = cellfun(@length, array);
out = zeros(prod(siz), nd);
for i = 1:nd
    s = ones(1, nd);
    s(i) = siz(i);
    x = reshape(array{i}, s);
    s = siz;
    s(i) = 1;
    out(:, i) = reshape(repmat(x, s), [], 1);
end
