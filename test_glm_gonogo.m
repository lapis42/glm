%% 1. Load data
clc; clearvars; close all;

% task parameter
DT = 0.001; % seconds

N_START = 10; % trial start, n bumps
RANGE_START = 3; % second
BIAS_START = 8;

N_END = 20; % trial end
RANGE_END = 10;
BIAS_END = 8;

N_REWARD = 15;
RANGE_REWARD = 7;
BIAS_REWARD = 8;

N_PUNISH = 10;
RANGE_PUNISH = 2;
BIAS_PUNISH = 1;

N_SPEED = 15; % n bumps
RANGE_SPEED = [-10, 40]; % cm/s

N_H = 10; % n bumps for spike history
RANGE_H = 0.2; % seconds
BIAS_H = 1;

% code parameter
PLOT = true;
CALC_PRM = false;

% file location
DATA_PATH = getenv('OneDrive');
EPHYS_PATH = fullfile(DATA_PATH, 'project\gonogo\data_ephys\');
ephys_file = dir(fullfile(EPHYS_PATH, '*data.mat'));
n_file = length(ephys_file);

% loading
i_file = 1; % for loop
load(fullfile(EPHYS_PATH, ephys_file(i_file).name));
disp([num2str(i_file), ' / ', num2str(n_file), ': ', ephys_file(i_file).name, ' loaded.']);



%% 2. Session data
session_range = Vr.timeNidq([1, end]);
session_duration = diff(session_range);
clip = @(t) t(t >= session_range(1) & t <= session_range(2)) - session_range(1);
time_bin = (0:DT:session_duration)';
n_bin = length(time_bin) - 1;

% behavior
vr_time = clip(Vr.timeNidq);
interp_time = @(value, unit) interp1(vr_time, double(value) / unit, time_bin(1:end-1), 'linear', 'extrap');
speed_coarse = basis.normal_filter([0; diff(Vr.position)], 0.5, 0.1) / 0.01 * 2.35;
speed = interp_time(speed_coarse, 1);

% task
time_nogo_start = histcounts(clip(Trial.timeStartNogoNidq), time_bin)';
time_nogo_end = histcounts(clip(Trial.timeEndNogoNidq), time_bin)';
time_go_start = histcounts(clip(Trial.timeStartGoNidq), time_bin)';
time_go_end = histcounts(clip(Trial.timeEndGoNidq), time_bin)';
time_reward = histcounts(clip(Trial.timeRewardNidq), time_bin)';
time_punish = histcounts(clip(Trial.timePunishmentNidq), time_bin)';

% bump
[start_base, start_time, start_func] = basis.log_cos(N_START, RANGE_START, DT, BIAS_START, false);
[end_base, end_time, end_func] = basis.log_cos(N_END, RANGE_END, DT, BIAS_END, false);
[reward_base, reward_time, reward_func] = basis.log_cos(N_REWARD, RANGE_REWARD, DT, BIAS_REWARD, false);
[punish_base, punish_time, punish_func] = basis.log_cos(N_PUNISH, RANGE_PUNISH, DT, BIAS_PUNISH, false);
[speed_basis, speed_time, speed_func] = basis.linear_cos(N_SPEED, RANGE_SPEED, 1, false);

% convolution
X_nogo_start = basis.conv(time_nogo_start, start_base);
X_nogo_end = basis.conv(time_nogo_end, end_base);
X_go_start = basis.conv(time_go_start, start_base);
X_go_end = basis.conv(time_go_end, end_base);
X_reward = basis.conv(time_reward, reward_base);
X_punish = basis.conv(time_punish, punish_base);
X_speed = speed_func(speed);


%% 3. Spike
% spike
i_cell = 20; % for loop
disp(Spike.posX(i_cell));

spike_time = clip(Spike.time{i_cell});
n_spike = length(spike_time);
spike_bin = histcounts(spike_time, time_bin)';
spike_rate = n_spike / session_duration;

% spike bump
[h_base, h_time] = basis.log_cos(N_H, [DT, RANGE_H], DT, BIAS_H, false);

% convolution
X_h = basis.conv(spike_bin, h_base, h_time(1)/DT);


%% 4. getting average response for parameter fitting (not necessary)
% 4.1. Firing rate by behavior
% coarse binning
time_bin_s = (0:session_duration)';
n_bin_s = length(time_bin_s);
ratio_time = floor(1 / DT);
coarse_bin = @(x) mean(reshape(x(1:floor(n_bin/ratio_time)*ratio_time), ratio_time, []))';

spike_s = coarse_bin(spike_bin) * ratio_time;
speed_s = coarse_bin(speed);
speed_edge = RANGE_SPEED(1):RANGE_SPEED(2) - 0.5;
[speed_mean, speed_sem, speed_bin] = func.group_stat(speed_s, spike_s, speed_edge);
speed_log = log(speed_mean / spike_rate);

if PLOT
    % plotting
    figure(11); clf;
    subplot(2, 1, 1);
    hold on;
    scatter(speed_s, spike_s, '.');
    errorbar(speed_bin, speed_mean, speed_sem);

    subplot(2, 1, 2);
    plot(speed_bin, speed_log);
end


%% 4.2. Firing rate by task
bin_size = 0.010; % seconds
filter_sigma = 0.100; % seconds
window = 5; % seconds
cut = 4 * filter_sigma / bin_size;

[spike_start_trial, time_task] = func.fast_align(clip(Trial.timeStartNidq), spike_time, ...
    bin_size, window + 4 * filter_sigma);
in_t = 1 + cut:length(time_task) - cut;
time_task = time_task(in_t);

trial_idx = (double(Trial.cueBcs) - 1) * 2 + double(Trial.resultBcs) + 1;
spike_task = func.group_stat2(spike_start_trial, trial_idx);
spike_task_conv = basis.normal_filter(spike_task, filter_sigma, bin_size);

task_log = log(spike_task_conv(in_t, :) / spike_rate);

if PLOT
    figure(12); clf;
    % rule-related response
    subplot(2, 1, 1);
    plot(time_task, spike_task_conv(in_t, :));
    title('rule');
    xlabel('time from start');

    % cue-related response
    subplot(2, 1, 2);
    plot(time_task, task_log);
    title('cue');
    xlabel('time from start');
end


%% 4.3. Average autocorrelogram
% get autocorrelation
[spc, time_spc] = func.cross_corr(spike_time, spike_time, DT, RANGE_H);
time_spc = time_spc((RANGE_H/DT)+2:end);
spc = spc((RANGE_H/DT)+2:end);
spc_log = log(spc / spike_rate + exp(-10));

if PLOT
    figure(13); clf;
    subplot(2, 1, 1);
    hold on;
    plot(time_spc, spc);
    plot(time_spc([1, end]), [spike_rate, spike_rate], 'k:');

    subplot(2, 1, 2);
    plot(time_spc, spc_log);
end


%% 5. Parameter fitting
if CALC_PRM
    start_base = start_func(time_task);
    speed_base = speed_func(speed_bin);
    roll_base = roll_func(roll_bin);
    
    % fitting weights by average response
    task_proj = pinv(start_base' * start_base) * start_base';
    w_rule0 = task_proj * task_log;
    w_cue0 = task_proj * cue_log;
    w_pchoice0 = task_proj * pchoice_log;
    w_choice0 = task_proj * choice_log;
    w_result0 = task_proj * result_log;
    w_speed0 = pinv(speed_base' * speed_base) * (speed_base' * speed_log);
    w_roll0 = pinv(roll_base' * roll_base) * (roll_base' * roll_log);
    w_h0 = pinv(h_base' * h_base) * (h_base' * spc_log);
    w_c0 = log(spike_rate);

    % rebuilding fitted kernel
    rule0 = start_base * w_rule0;
    cue0 = start_base * w_cue0;
    pchoice0 = start_base * w_pchoice0;
    choice0 = start_base * w_choice0;
    result0 = start_base * w_result0;
    speed0 = speed_base * w_speed0;
    roll0 = roll_base * w_roll0;
    h0 = h_base * w_h0;

    if PLOT
        % plotting
        figure(14); clf;
        subplot(4, 2, 1);
        hold on;
        plot(speed_bin, speed_log);
        plot(speed_bin, speed0);
        title('speed');

        subplot(4, 2, 2);
        hold on;
        plot(spc_log);
        plot(h0)
        title('spike history');

        subplot(4, 2, 3);
        hold on;
        plot(task_log);
        plot(rule0);
        title('rule');
        
        subplot(4, 2, 4);
        hold on;
        plot(cue_log);
        plot(cue0);
        title('cue');
        
        subplot(4, 2, 5);
        hold on;
        plot(pchoice_log);
        plot(pchoice0);
        title('pchoice');
        
        subplot(4, 2, 6);
        hold on;
        plot(choice_log);
        plot(choice0);
        title('choice');
        
        subplot(4, 2, 7);
        hold on;
        plot(result_log);
        plot(result0);
        title('result');
        
        subplot(4, 2, 8);
        hold on;
        plot(roll_log);
        plot(roll0);
        title('roll');
    end
end


%% 6. loss function
[X, prm] = func.normalize_add_constant({X_nogo_start, X_nogo_end, X_go_start, X_go_end, X_reward, X_punish, X_speed, X_h}, false);

if CALC_PRM
    prm0 = [w_c0; w_rule0(:); w_choice0(:); w_speed0; w_roll0; w_h0];
else
    w_c0 = log(spike_rate);
    prm0 = [w_c0; rand(prm.n_var, 1) - 0.5];
end
prm0_norm = [prm0(1) + prm.mean * prm0(2:end); prm0(2:end) .* prm.std'];

lfunc = @(w) loss.log_poisson_loss(w, X, spike_bin, DT, 1e-6);


%% optimization
algopts = {'algorithm','trust-region','Gradobj','on','Hessian','on', 'display', 'iter', 'maxiter', 100};
opts = optimset(algopts{:});
[prm1_norm, loss1, exitflag, output, grad, hessian] = fminunc(lfunc, prm0_norm, opts);


%% revert prm
prm1 = [prm1_norm(1) - (prm.mean ./ prm.std) * prm1_norm(2:end); prm1_norm(2:end) ./ prm.std'];
prm1_std_norm = sqrt(diag(inv(hessian)));
prm1_std = [prm1_std_norm(1); prm1_std_norm(2:end) ./ prm.std'];

%% plot
nogo_start1 = start_base * prm1(prm.index{2});
nogo_start1_std = start_base * prm1_std(prm.index{2});

nogo_end1 = end_base * prm1(prm.index{3});
nogo_end1_std = end_base * prm1_std(prm.index{3});

go_start1 = start_base * prm1(prm.index{4});
go_start1_std = start_base * prm1_std(prm.index{4});

go_end1 = end_base * prm1(prm.index{5});
go_end1_std = end_base * prm1_std(prm.index{5});

reward1 = reward_base * prm1(prm.index{6});
reward1_std = reward_base * prm1_std(prm.index{6});

punish1 = punish_base * prm1(prm.index{7});
punish1_std = punish_base * prm1_std(prm.index{7});

speed1 = speed_basis * prm1(prm.index{8});
speed1_std = speed_basis * prm1_std(prm.index{8});

h1 = h_base * prm1(prm.index{9});
h1_std = h_base * prm1_std(prm.index{9});

figure(15); clf;
subplot(4, 2, 1);
hold on;
plot(start_time, nogo_start1);
plot(start_time, go_start1);
title('start');

subplot(4, 2, 2);
hold on;
plot(end_time, nogo_end1);
plot(end_time, go_end1);
title('end');

subplot(4, 2, 3);
hold on;
plot(reward_time, reward1);
title('reward');

subplot(4, 2, 4);
hold on;
plot(punish_time, punish1);
title('punishment');

subplot(4, 2, 5);
hold on;
plot(speed_time, speed1);
title('speed');

subplot(4, 2, 6);
hold on;
% plot(h_time, h0)
plot(h_time, h1);
% errorbar(h_time, h1, h1_std);
title('spike history');