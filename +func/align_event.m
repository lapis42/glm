function [event_average, time_bin] = align_event(signal, event, window)
assert(length(signal) == length(event));
n_time = length(signal);


event_time = find(event > 0);
n_event = length(event_time);


time_bin = window(1):window(2);
n_bin = length(time_bin);
n_neg = sum(time_bin < 0);
n_pos = sum(time_bin > 0);


event_aligned = NaN(n_event, n_bin);
for i_event = 1:n_event
    if event_time(i_event) <= n_neg || event_time(i_event) > n_time - n_pos
        continue;
    else
        event_aligned(i_event, :) = signal(event_time(i_event) + time_bin);
    end
end

event_average = nanmean(event_aligned, 1)';
