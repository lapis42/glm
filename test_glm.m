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

% variables
%   sps: spike (size (500000, 1))
%   Stim: visual stimuli (size (50000, 1), range (-1, 1))
%   nkt: time range of stimulus filter (300 ms)
%   dtSp: time bin size for spike
%   dtStim: time bin size for stimuli


%% data conversion
session_duration = length(sps) * dtSp;

time_bin = (dtSp:dtSp:session_duration)';
n_bin = length(time_bin);

spike_binned = sps;
spike_time = find(spike_binned) * dtSp;
n_spike = length(spike_time);
spike_rate = mean(spike_binned) / dtSp;

k_true = flip(ggsim.k);
h_true = ggsim.ih;
w_c_true = ggsim.dc;

I_k_true = conv(Stim, k_true, 'full');
I_k_true = kron(I_k_true(1:end-nkt+1), ones(10, 1));
I_h_true = conv(spike_binned, h_true, 'full');
I_h_true = [0; I_h_true(1:end-161)];
I_total_true = I_k_true + I_h_true + w_c_true;


%% parameter
n_k = 8;
n_h = 8;
n_ht = 177;
n_kt = 300;


%% get average stimulus (looking backwards)
spike_coarse = sum(reshape(spike_binned, 10, []));
sta = func.align_event(Stim, spike_coarse, [-nkt+1, 0]);
sta = flipud(sta);


%% get average cross-correlogram
spc = func.cross_corr(spike_time, spike_time, dtSp, n_ht*dtSp);
spc = spc(n_ht+2:end);
spc_log = arrayfun(@log, spc / spike_rate + exp(-10));


%%  make bump
[k_base, k_time] = basis.log_cos(n_k, nkt, 10);
[h_base, h_time] = basis.log_cos(n_h, [1, n_ht+1], 10);

% fit bump to average response
w_k0 = pinv(k_base' * k_base) * (k_base' * sta);
w_h0 = pinv(h_base' * h_base) * (h_base' * spc_log);
w_c0 = log(spike_rate);

k0 = k_base * w_k0;
h0 = h_base * w_h0;

% plot
figure;
subplot(2, 2, 1);
hold on;
plot(k_true);
plot(k0);
plot(sta);

subplot(2, 2, 2);
plot(k_base);

subplot(2, 2, 3);
hold on;
plot(h_true);
plot(h0)
plot(spc_log);

subplot(2, 2, 4);
plot(h_base);


%% convolution
X_k_exp = basis.conv(Stim, k_base, k_time);
X_k = kron(X_k_exp, ones(10, 1));
X_h = basis.conv(spike_binned, h_base, h_time);


%% loss function
[X, prm] = func.normalize_add_constant({X_k, X_h}, true);
prm0 = [w_c0; w_k0; w_h0];
prm0_norm = [prm0(1) + prm.mean * prm0(2:end); prm0(2:end) .* prm.std'];

lfunc = @(w) loss.log_poisson_loss(w, X, spike_binned, dtSp);


%% optimization
algopts = {'algorithm','trust-region','Gradobj','on','Hessian','on', 'display', 'iter', 'maxiter', 100};
opts = optimset(algopts{:});
[prm1_norm, loss1, exitflag, output, grad, hessian] = fminunc(lfunc, prm0_norm, opts);


%% revert prm
prm1 = [prm1_norm(1) - (prm.mean ./ prm.std) * prm1_norm(2:end); prm1_norm(2:end) ./ prm.std'];


%% plot
k1 = k_base * prm1(prm.index{2});
h1 = h_base * prm1(prm.index{3});

I_total0 = X * prm0_norm;
I_total1 = X * prm1_norm;

figure;
subplot(2, 2, 1);
hold on;
plot(k_true);
plot(k0);
plot(k1);
legend({'true', 'k0', 'k1'});

subplot(2, 2, 2);
hold on;
plot(h_true);
plot(h0)
plot(h1);

subplot(2, 2, 3);
hold on;
plot(I_total_true(1:1000));
plot(I_total0(1:1000));
plot(I_total1(1:1000));

subplot(2, 2, 4);
hold on;
plot(exp(I_total_true(1:1000)));
plot(exp(I_total0(1:1000)));
plot(exp(I_total1(1:1000)));