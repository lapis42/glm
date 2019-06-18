%% 1. Load data
clc; clearvars; close all;


% task parameter
DT = 0.001; % seconds

N_TASK = 10; % n bumps
RANGE_TASK = 5; % second

N_SPEED = 10; % n bumps
RANGE_SPEED = 50; % cm/s
BIAS_SPEED = 10;

N_ROLL = 10;
RANGE_ROLL = 20; % cm/s

N_H = 10; % n bumps for spike history
RANGE_H = 0.2; % seconds
BIAS_H = 1;


% code parameter
PLOT = true;
CALC_PRM = true;


% file parameter
DATA_PATH = getenv('OneDrive');
EPHYS_PATH = fullfile(DATA_PATH, 'project\ruleswitch\data_ephys\');
ephys_file = dir(fullfile(EPHYS_PATH, '*data.mat'));
n_file = length(ephys_file);


% loading
i_file = 32; % for loop
load(fullfile(EPHYS_PATH, ephys_file(i_file).name), 'Spike', 'Trial', 'Vr');
disp([ephys_file(i_file).name, ' loaded.']);




%% 2. Session data
session_range = Vr.timeImec([1, end]);
session_duration = diff(session_range);
clip = @(t) t(t >= session_range(1) & t <= session_range(2)) - session_range(1);
time_bin = (0:DT:session_duration)';
n_bin = length(time_bin) - 1;

% behavior
vr_time = clip(Vr.timeImec);
interp_time = @(value, unit) interp1(vr_time, double(value) / unit, time_bin(1:end-1), 'linear', 'extrap');
speed = interp_time(Vr.ball_speed, 1); % ~ 39 MB; sparse matrix didn't save a lot...
roll = interp_time(Vr.roll, 10); % ~39 MB
angle = interp_time(Vr.angle, 1);

% task
time_start = clip(Trial.timeStartVr);
time_end = clip(Trial.timeResultImec);
time_delay = clip(Trial.timeDelayVr);

% bump
[task_base, task_time, task_func] = basis.linear_cos(N_TASK, RANGE_TASK, DT, false);
[~, ~, speed_func] = basis.log_cos(N_SPEED, RANGE_SPEED, 1, BIAS_SPEED, false);
[~, ~, roll_func] = basis.linear_cos(N_ROLL, RANGE_ROLL, 1, false);

% convolution
[rule_bin, choice_bin, cue_bin] = deal(zeros(n_bin, 2));
for i_task = 1:2
    rule_bin(:, i_task) = histcounts(time_start(Trial.task == i_task), time_bin);
    choice_bin(:, i_task) = histcounts(time_start(Trial.choice == i_task), time_bin);
    cue_bin(:, i_task) = histcounts(time_start(Trial.cue == i_task), time_bin);
end

X_rule = basis.conv(rule_bin, task_base, task_time(1)/DT);
X_choice = basis.conv(choice_bin, task_base, task_time(1)/DT);
X_cue = basis.conv(cue_bin, task_base, task_time(1)/DT);

X_speed = speed_func(speed);
X_roll = roll_func(roll);


%% 3. Spike
% spike
i_cell = 21; % for loop
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
if CALC_PRM
    % coarse binning
    time_bin_s = (0:session_duration)';
    n_bin_s = length(time_bin_s);

    ratio_time = floor(1 / DT);
    coarse_bin = @(x) mean(reshape(x(1:floor(n_bin/ratio_time)*ratio_time), ratio_time, []))';

    spike_s = coarse_bin(spike_bin) * ratio_time;
    speed_s = coarse_bin(speed);
    roll_s = coarse_bin(roll);
    angle_s = coarse_bin(angle);

    speed_edge = (0:RANGE_SPEED+1) - 0.5;
    roll_edge = -RANGE_ROLL:RANGE_ROLL+1 - 0.5;
    angle_edge = (-pi:pi/20:pi+pi/20) - pi/40;

    [speed_mean, speed_sem, speed_bin] = func.group_stat(speed_s, spike_s, speed_edge);
    [roll_mean, roll_sem, roll_bin] = func.group_stat(roll_s, spike_s, roll_edge);
    [angle_mean, angle_sem, angle_bin] = func.group_stat(angle_s, spike_s, angle_edge);

    speed_log = log(speed_mean / spike_rate);
    roll_log = log(roll_mean / spike_rate);
    angle_log = log(angle_mean / spike_rate);

    if PLOT
        % plotting
        figure(11); clf;
        subplot(3, 2, 1);
        hold on;
        scatter(speed_s, spike_s, '.');
        errorbar(speed_bin, speed_mean, speed_sem);

        subplot(3, 2, 2);
        plot(speed_bin, speed_log);

        subplot(3, 2, 3);
        hold on;
        scatter(roll_s, spike_s, '.');
        errorbar(roll_bin, roll_mean, roll_sem);

        subplot(3, 2, 4);
        plot(roll_bin, roll_log);

        subplot(3, 2, 5);
        hold on;
        scatter(angle_s, spike_s, '.');
        errorbar(angle_bin, angle_mean, angle_sem);

        subplot(3, 2, 6);
        plot(angle_bin, angle_log);
    end
end


%% 4.2. Firing rate by task
if CALC_PRM
    bin_size = 0.010; % seconds
    filter_sigma = 0.100; % seconds
    window = 5; % seconds
    cut = 4 * filter_sigma / bin_size;
    
    [spike_start_trial, time_task] = func.fast_align(time_start, spike_time, ...
        bin_size, window + 4 * filter_sigma);
    spike_end_trial = func.fast_align(time_end, spike_time, ...
        bin_size, window + 4 * filter_sigma);
    spike_delay_trial = func.fast_align(time_delay, spike_time, ...
        bin_size, window + 4 * filter_sigma);
    in_t = 1 + cut:length(time_task) - cut;
    time_task = time_task(in_t);

    spike_rule = func.group_stat2(spike_start_trial, Trial.task);
    spike_cue = func.group_stat2(spike_start_trial, Trial.cue);
    spike_pchoice = func.group_stat2(spike_delay_trial, Trial.pChoice);
    spike_choice = func.group_stat2(spike_start_trial, Trial.choice);
    spike_result = func.group_stat2(spike_end_trial, Trial.result + 1);

    spike_rule_conv = basis.normal_filter(spike_rule, filter_sigma, bin_size);
    spike_cue_conv = basis.normal_filter(spike_cue, filter_sigma, bin_size);
    spike_pchoice_conv = basis.normal_filter(spike_pchoice, filter_sigma, bin_size);
    spike_choice_conv = basis.normal_filter(spike_choice, filter_sigma, bin_size);
    spike_result_conv = basis.normal_filter(spike_result, filter_sigma, bin_size);
    
    rule_log = log(spike_rule_conv(in_t, :) / spike_rate);
    cue_log = log(spike_cue_conv(in_t, :) / spike_rate);
    pchoice_log = log(spike_pchoice_conv(in_t, :) / spike_rate);
    choice_log = log(spike_choice_conv(in_t, :) / spike_rate);
    result_log = log(spike_result_conv(in_t, :) / spike_rate);

    if PLOT
        figure(12); clf;
        % rule-related response
        subplot(3, 2, 1);
        plot(time_task, rule_log);
        title('rule');
        xlabel('time from start');

        % cue-related response
        subplot(3, 2, 2);
        plot(time_task, cue_log);
        title('cue');
        xlabel('time from start');

        % pchoice-related response
        subplot(3, 2, 3);
        plot(time_task, pchoice_log);
        title('p-choice');
        xlabel('time from delay');

        % choice-related response
        subplot(3, 2, 4);
        plot(time_task, choice_log);
        title('choice');
        xlabel('time from start');

        % reward-related response
        subplot(3, 2, 5);
        plot(time_task, result_log);
        title('result');
        xlabel('time from trial end');
    end
end


%% 4.3. Average autocorrelogram
if CALC_PRM
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
end


%% 5. Parameter fitting
if CALC_PRM
    task_base = task_func(time_task);
    speed_base = speed_func(speed_bin);
    roll_base = roll_func(roll_bin);
    
    % fitting weights by average response
    task_proj = pinv(task_base' * task_base) * task_base';
    w_rule0 = task_proj * rule_log;
    w_cue0 = task_proj * cue_log;
    w_pchoice0 = task_proj * pchoice_log;
    w_choice0 = task_proj * choice_log;
    w_result0 = task_proj * result_log;
    w_speed0 = pinv(speed_base' * speed_base) * (speed_base' * speed_log);
    w_roll0 = pinv(roll_base' * roll_base) * (roll_base' * roll_log);
    w_h0 = pinv(h_base' * h_base) * (h_base' * spc_log);
    w_c0 = log(spike_rate);

    % rebuilding fitted kernel
    rule0 = task_base * w_rule0;
    cue0 = task_base * w_cue0;
    pchoice0 = task_base * w_pchoice0;
    choice0 = task_base * w_choice0;
    result0 = task_base * w_result0;
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
        plot(rule_log);
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
[X, prm] = func.normalize_add_constant({X_rule, X_choice, X_speed, X_roll, X_h}, false);
prm0 = [w_c0; w_rule0(:); w_choice0(:); w_speed0; w_roll0; w_h0];
prm0_norm = [prm0(1) + prm.mean * prm0(2:end); prm0(2:end) .* prm.std'];

lfunc = @(w) loss.log_poisson_loss(w, X, spike_bin, DT);


%% optimization
algopts = {'algorithm','trust-region','Gradobj','on','Hessian','on', 'display', 'iter', 'maxiter', 100};
opts = optimset(algopts{:});
[prm1_norm, loss1, exitflag, output, grad, hessian] = fminunc(lfunc, prm0_norm, opts);


%% revert prm
prm1 = [prm1_norm(1) - (prm.mean ./ prm.std) * prm1_norm(2:end); prm1_norm(2:end) ./ prm.std'];
prm1_std_norm = sqrt(diag(inv(hessian)));
prm1_std = [prm1_std_norm(1); prm1_std_norm(2:end) ./ prm.std'];

%% plot
rule1 = task_base * reshape(prm1(prm.index{2}), [], 2);
rule1_std = task_base * reshape(prm1_std(prm.index{2}), [], 2);

choice1 = task_base * reshape(prm1(prm.index{3}), [], 2);
choice1_std = task_base * reshape(prm1_std(prm.index{3}), [], 2);
% cue1 = task_base * reshape(prm1(prm.index{4}), [], 2);
% cue1_std = task_base * reshape(prm1_std(prm.index{4}), [], 2);

speed1 = speed_base * prm1(prm.index{4});
speed1_std = speed_base * prm1_std(prm.index{4});

roll1 = roll_base * prm1(prm.index{5});
roll1_std = roll_base * prm1_std(prm.index{5});

h1 = h_base * prm1(prm.index{6});
h1_std = h_base * prm1_std(prm.index{6});

figure(15); clf;
subplot(4, 2, 1);
hold on;
plot(time_task, rule0);
plot(time_task, rule1);
% for i_task = 1:2
%     errorbar(time_task(1:10:end), rule1(1:10:end, i_task), rule1_std(1:10:end, i_task));
% end
title('rule');

subplot(4, 2, 2);
hold on;
plot(time_task, choice0);
plot(time_task, choice1);
% for i_task = 1:2
%     errorbar(time_task(1:10:end), choice1(1:10:end, i_task), choice1_std(1:10:end, i_task));
% end
title('choice');

% subplot(4, 2, 3);
% hold on;
% plot(time_task, cue0);
% plot(time_task, cue1);
% % for i_task = 1:2
% %     errorbar(time_task(1:10:end), cue1(1:10:end, i_task), cue1_std(1:10:end, i_task));
% % end
% title('cue');

subplot(4, 2, 4);
hold on;
plot(speed_bin, speed0);
errorbar(speed_bin, speed1, speed1_std);
title('speed');

subplot(4, 2, 5);
hold on;
plot(roll_bin, roll0);
errorbar(roll_bin, roll1, roll1_std);
title('roll');

subplot(4, 2, 6);
hold on;
plot(h_time, h0)
errorbar(h_time, h1, h1_std);
title('spike history');