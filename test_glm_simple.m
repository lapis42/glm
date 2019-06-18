% Log linear model
%
% firing_rate(t) = 
%   exp( conv(stimulus, k)(t) + conv(spike_history, k)(t) + constant )
%
% The kernel k and h can be decomposed into multiple subkernels like cosine
% bumps.
%       k = k1 * w1 + k2 * w2 + k3 * w3
%
% Distributivity can be applied to convolution.
%       conv( f, (g + h) ) = conv( f, g ) + conv ( f, h )
%
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


%% load data
clc; clearvars; close all;
load('glm_data.mat');

% Used variable
% Stim: stimulus (size 50000 x 1)
% sps: spike (size 500000 x 1)
% dtSp: spike bin size (0.001 s)

% k_true: true kernel for stimuli
% h_true: true kernel for spike history
% w_c_true: true constant
% I_total_true: true (X*w)

%%  make bump
% log_cos( number of bumps, bin window, bias(lower-loglinear, higher-linear) )
n_k_bump = 8;
n_k_time = 30; % 300 ms
n_h_bump = 8;
n_h_time = 177; % 177 ms

[k_base, k_time] = basis.log_cos(n_k_bump, n_k_time, 10);
[h_base, h_time] = basis.log_cos(n_h_bump, [1, n_h_time+1], 10);


%% convolution
X_k_coarse = basis.conv(Stim, k_base, k_time);
X_k = kron(X_k_coarse, ones(10, 1)); % matching time bin size difference
X_h = basis.conv(sps, h_base, h_time);


%% loss function
% normalize_add_constant( {cell arrays to combine}, normalize? )
[X, prm] = func.normalize_add_constant({X_k, X_h}, true);

% log_poisson_loss( weight, design matrix, spike, spike bin size )
lfunc = @(w) loss.log_poisson_loss(w, X, sps, dtSp);

% initial weight
prm0_norm = rand(size(X, 2), 1) - 0.5;


%% optimization
algopts = {'algorithm','trust-region','Gradobj','on','Hessian','on', 'display', 'iter', 'maxiter', 100};
opts = optimset(algopts{:});
[prm1_norm, loss1, exitflag, output, grad, hessian] = fminunc(lfunc, prm0_norm, opts);
prm1_std_norm = sqrt(diag(inv(hessian)));


%% revert prm
prm1 = [prm1_norm(1) - (prm.mean ./ prm.std) * prm1_norm(2:end); prm1_norm(2:end) ./ prm.std'];
prm1_std = [prm1_std_norm(1); prm1_std_norm(2:end) ./ prm.std'];

%% plot
k1 = k_base * prm1(prm.index{2});
k1_std = k_base * prm1_std(prm.index{2});

h1 = h_base * prm1(prm.index{3});
h1_std = h_base * prm1_std(prm.index{3});

I_total1 = X * prm1_norm;

figure;
subplot(2, 2, 1);
hold on;
plot(k_true);
errorbar(1:n_k_time, k1, k1_std);
legend({'true', 'k1'});

subplot(2, 2, 2);
hold on;
plot(h_true);
errorbar(1:n_h_time, h1, h1_std);

subplot(2, 2, 3);
hold on;
plot(I_total_true(1:1000));
plot(I_total1(1:1000));

subplot(2, 2, 4);
hold on;
plot(exp(I_total_true(1:1000)));
plot(exp(I_total1(1:1000)));