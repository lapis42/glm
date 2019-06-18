import os
import time

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


def make_cos_bump(n_bump, bin_size=100, b=10, range_until=None):
    if range_until is None:
        range_until = bin_size

    bump_range = np.array([0, bin_size * (1 - 1.5 / n_bump)])

    def log_func(x): return np.log(x + b)

    def exp_func(x): return np.exp(x) - b

    range_log = log_func(bump_range)
    gap_log = np.diff(range_log) / (n_bump - 1)

    def cos_func(x, peak): return (np.cos(np.max([-np.pi,
                                                  np.min([np.pi,
                                                          (x - peak) * np.pi / (2 * gap_log)])])) + 1) / 2
    cos_func_vec = np.vectorize(cos_func)

    bin_log = np.tile(log_func(np.arange(range_until))[:, np.newaxis], n_bump)
    peak_log = np.tile(np.arange(
        range_log[0], range_log[1] + gap_log, gap_log)[:, np.newaxis], range_until).T

    bump = cos_func_vec(bin_log, peak_log)
    bump /= np.sqrt(np.sum(bump**2, 0))

    return bump


def main():
        # parameter
    n_k = 8
    n_h = 8
    n_ht = 177

    # load spike data
    data = sio.loadmat('glm_data.mat', squeeze_me=True)
    sps = data['sps']
    Stim = data['Stim']
    dtSp = data['dtSp']

    # dimension
    n_kt = data['nkt']
    n_sps = len(sps)
    n_stim = len(Stim)
    ratio = int(n_sps / n_stim)

    # get sta
    sps_coarse = np.sum(sps.reshape((-1, ratio)), 1)
    spike_time = np.where(sps_coarse > 0)[0]
    n_spike = len(spike_time)
    stim_aligned = np.zeros((n_spike, n_kt))
    for i_spike in range(n_spike):
        if spike_time[i_spike] < n_kt:
            stim_aligned[i_spike, n_kt-spike_time[i_spike] -
                         1:] = Stim[0:spike_time[i_spike]+1]
        else:
            stim_aligned[i_spike, :] = Stim[(
                spike_time[i_spike]-n_kt+1):(spike_time[i_spike]+1)]
    sta = np.flip(np.mean(stim_aligned, 0))

    # get cross-correlogram
    spike_time = np.where(sps > 0)[0]
    n_spike = len(spike_time)
    spike_aligned = np.zeros((n_spike, n_ht))
    for i_spike in range(n_spike):
        if spike_time[i_spike] > n_sps - n_ht - 1:
            spike_aligned[i_spike, 0:n_sps -
                          spike_time[i_spike] - 1] = sps[spike_time[i_spike]+1:]
        else:
            spike_aligned[i_spike, :] = sps[spike_time[i_spike] +
                                            1:spike_time[i_spike]+1+n_ht]
    stc = np.log(np.mean(spike_aligned, 0) / np.mean(sps) + np.exp(-10))

    # make bump
    k_base = make_cos_bump(n_k, n_kt, 10)
    h_base = make_cos_bump(n_h, np.floor(n_ht*2/3), 20, n_ht)

    # calc starting weights
    w_k0 = np.linalg.pinv(k_base.T @ k_base) @ (k_base.T @ sta)
    w_h0 = np.linalg.pinv(h_base.T @ h_base) @ (h_base.T @ stc)

    # calc starting kernels
    k0 = k_base @ w_k0
    h0 = h_base @ w_h0
    c0 = np.log(np.mean(sps))

    # convolution
    k_base_conv = np.zeros((n_sps, n_k))
    for i_k in range(n_k):
        k_base_conv_temp = np.convolve(Stim, k_base[:, i_k])
        k_base_conv[:, i_k] = np.kron(
            k_base_conv_temp[0:-n_kt+1], np.ones((1, ratio)))

    h_base_conv = np.zeros((n_sps, n_h))
    for i_h in range(n_h):
        h_base_conv_temp = np.convolve(sps, h_base[:, i_h])
        h_base_conv[:, i_h] = np.append(0, h_base_conv_temp[0:-n_ht])

    X = np.concatenate([k_base_conv, h_base_conv], 1).astype(np.float32)
    n_X = X.shape[1]
    y = sps.astype(np.float32)

    # make model
    prm0 = np.append(w_k0, w_h0)
    model = keras.models.Sequential([
        keras.layers.Dense(1, input_shape=[n_X],
                           kernel_initializer=keras.initializers.Constant(
                               value=prm0),
                           bias_initializer=keras.initializers.Constant(value=c0),
                           kernel_regularizer=keras.regularizers.l2(1e-4))])
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.0027, beta1=0.89, beta2=0.9999)
    model.compile(optimizer=optimizer,
                  loss=tf.nn.log_poisson_loss)

    #model.compile(optimizer=optimizer,
    #              loss=keras.metrics.poisson)
    early_stop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-5)

    # run model
    t = time.time()
    history = model.fit(X, y,
                        batch_size=400000, epochs=50, validation_split=0.2, callbacks=[early_stop])
    elapsed_time = time.time() - t
    print('Elapsed time: ', elapsed_time)

    #######################################
    # Result
    # AdamOptimizer, tf.nn.log_poisson_loss, batch_size=total: 14 s
    # AdamOptimizer, tf.metrics.poisson with exponential layer: 14 s

    #######################################


    # unwrap weights
    w_k1 = model.get_weights()[0][0:8]
    w_h1 = model.get_weights()[0][8:]
    w_c1 = model.get_weights()[1][0]

    # calc starting kernels
    k1 = k_base @ w_k1
    h1 = h_base @ w_h1

    # plot
    _, ax = plt.subplots(3)
    ax[0].plot(sta, label='sta')
    ax[0].plot(k0, label='k0')
    ax[0].plot(k1, label='k1')
    ax[0].legend(loc=1)

    ax[1].plot(stc, label='stc')
    ax[1].plot(h0, label='h0')
    ax[1].plot(h1, label='h1')
    ax[1].legend(loc=1)

    ax[2].plot(history.history["loss"], label='Train loss')
    ax[2].plot(history.history["val_loss"], label='Validation loss')
    ax[2].legend(loc=1)
    plt.show()


if __name__ == "__main__":
    main()
