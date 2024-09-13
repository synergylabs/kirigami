import time
from datetime import timedelta as td
import librosa
import numpy as np
import scipy
from init_config import *


def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x, ):
    return librosa.core.db_to_amplitude(x, ref=1.0)


def recalibrate_mask_data(background_stft_mask_data):
    # Convert to noise stft to Dbs
    noise_stft_db = _amp_to_db(background_stft_mask_data)
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    return noise_stft_db, mean_freq_noise, std_freq_noise, noise_thresh


def compute_mask(prev_profile, new_bg_noise):
    bg_fft_mask = np.hstack((prev_profile, new_bg_noise))
    noise_stft_db, mean_freq_noise, std_freq_noise, noise_thresh = recalibrate_mask_data(bg_fft_mask)
    return noise_stft_db, mean_freq_noise, std_freq_noise, noise_thresh


def apply_mask_spectrogram(audio_stft, audio_stft_complex, noise_thresh, mean_freq_noise, smoothing_filter):
    #noise_stft_db, mean_freq_noise, std_freq_noise, noise_thresh = compute_mask(audio_stft, noise_fft_data)
    sig_stft_db = _amp_to_db(audio_stft)
    mask_gain_dB = np.min(_amp_to_db(audio_stft))

    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T

    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh

    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease

    # sig_stft_img = audio_stft_complex

    # mask the signal
    sig_stft_db_masked = (
            sig_stft_db * (1 - sig_mask)
            + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )
    return _db_to_amp(sig_stft_db_masked)