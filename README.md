# GLM for neural data analysis


## Note
### Log linear model
Firing rate ($r(t)$) at time $t$ is modeled as,

$$
r(t) = e^{ \sum_{i}(\mathbf{k_i} _\ast x_i)(t) + c }
$$

The kernel k and h can be decomposed into multiple subkernels like boxcar functions or cosine bumps.

$$
\mathbf{k} = \mathbf{k_1} * w_1 + \mathbf{k_2} * w_2 + \mathbf{k_3} * w_3
$$

Distributivity can be applied to convolution.

$$
f \ast (g + h) = f \ast g + f \ast h
$$

% firing_rate(t) = 
%   exp( conv(stim, k1*w1)(t) + conv(stim, k2*w2)(t) + conv(stim(t), k3*w3)
%        + conv(sph, h1*w4)(t) + conv(sph, h2*w5)(t) + const )
%
% Associativity with scalar multiplication for any real number a.
%       a * conv( f, g ) = conv ( a*f, g )
%
% firing_rate(t) =
%   exp( conv(stim, k1)(t) * w1 + conv(stim, k2)(t) * w2 ... + const )
%
% <=> firing_rate(t) = exp( [1, conv(stim, k1)(t), conv(stim, k2)(t) ... ] * w
%   where w = [const, w1, w2, ... ]'
%
% <=> firing_rate = exp( X * w )
%   where X is a combined matrix convolved by kernels (size: n_sample x
%   (1+n_kernel)). The first column is ones for constant weight.
%   w is a weight vector for these kernels.


% Poisson negative log-likelihood
%
% p(spike(t) | firing_rate(t)*dt)
%   = exp(-firing_rate(t)*dt) * (firing_rate(t)*dt)^(spike(t)) / (spike(t)!)
%
% -log( p(t) )
%   = firing_rate(t)*dt - spike(t) * log(firing_rate(t)*dt) + log(spike(t)!)
%
% -log( p )
%   = sum(firing_rate(t)) * dt - sum(spike(t) * log(firing_rate(t)*dt) + else
%   = sum(firing_rate(t)) * dt - sum(spike(t) * log(firing_rate(t)) + else2
%   = sum(exp(X * w)) * dt - spike' * (X * w) + else2
%
% To maximize p, we should minimize negative log-likelihood.
%
% Gradient for w = X' * (exp(X * w)*dt - spike)
% Hessian for w = X' * diag(exp(X * w)*dt) * X

