%% 1. Load data
clc; clearvars; close all;
load('glm_data.mat');
% variables
%   spike_bin: spike count per time bin (size 27871 x 1)
%   space_bin: location per time bin (27871 x 1)
%   speed_bin: speed at cm/s per time bin (27871 x 1)
%   dt: time bin size (50 ms)


%% 2. Parameter
SPACE_N = 36;
SPACE_RANGE = [0, 105];
SPEED_N = 19;
SPEED_RANGE = [3, 39];


%% 3. Build design matrix
space_func = basis.boxcar(SPACE_N, SPACE_RANGE);
speed_func = basis.boxcar(SPEED_N, SPEED_RANGE);
X_space = space_func(space_bin); % size 27871 x 36
X_speed = speed_func(speed_bin); % size 27871 x 19
y = spike_bin;


%% 4. Run GLM
out0 = glm.cvglm({X_space, X_speed}, y);
out1 = glm.cvglm({X_space, X_speed}, y, 'order', 1);


%% 5. Plot result
fig = figure(1);
subplot(231);
plot(out0.w(out0.prm.index{2}));
subplot(232);
plot(out0.w(out0.prm.index{3}));
subplot(233);
imagesc(log(out0.lambdas), log(out0.lambdas), reshape(out0.cv_deviance_mean, 10, 10));
subplot(234);
plot(out1.w(out1.prm.index{2}));
subplot(235);
plot(out1.w(out1.prm.index{3}));
subplot(236);
imagesc(log(out1.lambdas), log(out1.lambdas), reshape(out1.cv_deviance_mean, 10, 10));

