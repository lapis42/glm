function out = ndgrid(array,nd)
%NDGRID Rectangular grid in N-D space
%   OUT = NDGRID(ARRAY,ND) replicates the array and generates n-dimensional
%   grid.

k = numel(array);
out = zeros(k^nd, nd);
siz(1:nd) = k;

for i = 1:nd
    s = ones(1, nd);
    s(i) = k;
    x = reshape(array, s);
    s = siz;
    s(i) = 1;
    out(:, i) = reshape(repmat(x, s), [], 1);
end
