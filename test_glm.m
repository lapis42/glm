%% 1. Load data
clc; clearvars; close all;
load('glm_data.mat');
% variables
%   spike_bin: spike count per time bin (size 27871 x 1)
%   space_bin: location per time bin (27871 x 1)
%   speed_bin: speed at cm/s per time bin (27871 x 1)
%   dt: time bin size (50 ms)


%% 2. Parameter
SPACE_N = 22;
SPACE_RANGE = [0, 105];
SPEED_N = 20;
SPEED_RANGE = [3, 41];


%% 3. Build design matrix
[space_func, ~, ~, space_peak] = basis.boxcar(SPACE_N, SPACE_RANGE);
[speed_func, ~, ~, speed_peak] = basis.boxcar(SPEED_N, SPEED_RANGE);
X_space = space_func(space_bin); % size 27871 x 36
X_speed = speed_func(speed_bin); % size 27871 x 19
y = spike_bin;


%% 4. Run GLM
% Run ridge regression with a fixed lambda for all variables
out00 = glm.glm({X_space, X_speed}, y, 'lambda', 1);

% Run ridge regression with fixed lambdas where lambdas are given separately for each variable
out01 = glm.glm({X_space, X_speed}, y, 'lambda', {0.1, 1});

% Run rigde regression with grid-search for each variable
out02 = glm.glm({X_space, X_speed}, y, 'lambda', [0.01, 0.1, 1, 10]);

% Run basic cross-validation with grid-search
out0 = glm.cvglm({X_space, X_speed}, y, 'grid', 'on');

% Run regrssion with first-order regularization. Separately search lambdas for each variable (no-grid search)
out1 = glm.cvglm({X_space, X_speed}, y, 'order', 1, 'grid', 'each');

% Run regression with second-order regularization with custom lambda range for each variable.
lambda_range = {[1e-2, 1e3], [1, 1e5]};
n_lambda = [50, 100];
out2 = glm.cvglm({X_space, X_speed}, y, ...
    'order', 2, ...
    'lambda_range', lambda_range, ...
    'n_lambda', n_lambda, ...
    'grid', 'each');


%% 5. Plot result
fig = figure(1);
subplot(331);
plot(space_peak, out0.w{1});
subplot(332);
plot(speed_peak, out0.w{2});
subplot(333);
imagesc(log10(out0.lambda{1}), log10(out0.lambda{2}), out0.cv_deviance_mean);
subplot(345);
plot(space_peak, out1.w{1});
subplot(346);
plot(log10(out1.lambda{1}), out1.cv_deviance_mean{1});
subplot(347);
plot(speed_peak, out1.w{2});
subplot(348);
plot(log10(out1.lambda{2}), out1.cv_deviance_mean{2});
subplot(349);
plot(space_peak, out2.w{1});
subplot(3, 4, 10);
plot(log10(out2.lambda{1}), out2.cv_deviance_mean{1});
subplot(3, 4, 11);
plot(speed_peak, out2.w{2});
subplot(3, 4, 12);
plot(log10(out2.lambda{2}), out2.cv_deviance_mean{2});
