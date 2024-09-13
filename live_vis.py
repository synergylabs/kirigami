import json
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy
from scipy.signal import spectrogram, stft
from scipy.io import wavfile
import matplotlib.animation as animation
import torch

from cmsisdsp import arm_rfft_instance_q15, arm_rfft_init_q15, arm_rfft_q15, arm_cmplx_mag_q15
from cmsisdsp import arm_float_to_q15

from kirigami_filters import background_subtraction_utils as bgutils

from init_config import *
from kirigami_filters.filters import background_detection_filter, kirigami_filter, kirigami_filter_reverse_fft

# Function to load configuration from config.json


def _get_edge_fft_config():
    return enable_edge_fft

def list_microphones():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    MICROPHONES_LIST = []
    MICROPHONES_DESCRIPTION = []
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            desc = "# %d - %s" % (i, p.get_device_info_by_host_api_device_index(0, i).get('name'))
            MICROPHONES_DESCRIPTION.append(desc)
            MICROPHONES_LIST.append(i)

    output = []
    output.append("=== Available Microphones: ===")
    output.append("\n".join(MICROPHONES_DESCRIPTION))
    output.append("======================================")
    return "\n".join(output), MICROPHONES_DESCRIPTION, MICROPHONES_LIST


###########################
# Check Microphone
###########################
print("=====")
print("1 / 2: Checking Microphones... ")
print("=====")

desc, mics, indices = list_microphones()

print(desc)

if len(mics) == 0:
    print("Error: No microphone found.")
    exit()

#############
# Read Command Line Args
#############
MICROPHONE_INDEX = indices[0]
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mic", help="Select which microphone / input device to use")
args = parser.parse_args()
try:
    if args.mic:
        MICROPHONE_INDEX = int(args.mic)
        print("User selected mic: %d" % MICROPHONE_INDEX)
    else:
        mic_in = input("Select microphone [%d]: " % MICROPHONE_INDEX).strip()
        if (mic_in != ''):
            MICROPHONE_INDEX = int(mic_in)
except:
    print("Invalid microphone")
    exit()

# Find description that matches the mic index
mic_desc = ""
for k in range(len(indices)):
    i = indices[k]
    if i == MICROPHONE_INDEX:
        mic_desc = mics[k]
print("Using mic: %s" % mic_desc)

def calculate_q15_spectrogram(audio_samples, rfft_instance_q15, window_size=256, step_size=128):

    # Convert the audio to q15
    audio_samples_q15 = arm_float_to_q15(audio_samples)

    # Convert the q15 to 4096 range
    audio_samples_q15 = np.interp(audio_samples_q15, (audio_samples_q15.min(), audio_samples_q15.max()),
                                  (0, 4096)).astype(np.int32)

    # audio_samples_q15 = np.interp(audio_samples_q15, (audio_samples_q15.min(), audio_samples_q15.max()),
    #                               (-2048, 2048)).astype(np.int32)

    # Calculate the number of windows
    number_of_windows = int(1 + (len(audio_samples) - window_size) // step_size)

    # Calculate the FFT Output size
    num_fft_bins = int(window_size // 2 + 1)

    # Create an empty array to hold the Spectrogram
    spectrogram_q15 = np.empty((number_of_windows, num_fft_bins))

    start_index = 0
    # Apply hanning window and apply fft
    for index in range(number_of_windows):
        # Take the window from the waveform.
        audio_window_q15 = audio_samples_q15[start_index:start_index + window_size]

        # Calculate the FFT
        rfft_q15 = arm_rfft_q15(rfft_instance_q15, audio_window_q15)


        # Take the absolute value of the FFT and add to the Spectrogram.
        rfft_mag_q15 = arm_cmplx_mag_q15(rfft_q15)[:num_fft_bins]

        spectrogram_q15[index] = rfft_mag_q15

        # Increase the start index of the window by the overlap amount.
        start_index += step_size

    return spectrogram_q15


def cmsis_feature(wavs):
    rfft_instance_q15 = arm_rfft_instance_q15()
    status = arm_rfft_init_q15(rfft_instance_q15, 256, 0, 1)
    stft = calculate_q15_spectrogram(wavs.astype(np.float32), rfft_instance_q15, 256, 128)
    return stft


def prepare_plot(interpolation="nearest", vmax=20, vmin=0, config="Kirigami Demo"):
    fig, ax = plt.subplots(5, figsize=(20, 10))

    # Set the window title
    fig.canvas.manager.set_window_title(config)

    plt.title(config)

    # fig, (axis1, axis2, axis3, axis4) = plt.subplots(4)
    # fig, (axis1, axis2, axis3, axis4) = plt.subplots(nrows=4, ncols=1)
    plt.subplots_adjust(hspace=1.2)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_title('Raw Audio')
    ax[0].set_ylim(-AMPLITUDE, AMPLITUDE)

    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title('Original STFT')
    # axis2.set_ylim(0, 1)

    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Frequency')
    ax[2].set_title('Background Detection')

    ax[3].set_xlabel('Time')
    ax[3].set_ylabel('Frequency')
    ax[3].set_title('Background Masked Audio')

    ax[4].set_xlabel('Time')
    ax[4].set_ylabel('Frequency')
    ax[4].set_title('Speech Filtered Audio')

    x = np.arange(0, CHUNK * CHANNELS)
    x_fft = np.linspace(0, RATE, CHUNK)

    line, = ax[0].plot(x, np.random.randn(CHUNK * CHANNELS))
    # Create empty plot for the spectrogram
    # im = axis2.imshow(np.zeros((NFFT // 2 + 1, 100)), aspect='auto', cmap='inferno',
    #                extent=[0, 1, 0, RATE / 2], origin='lower')

    if enable_edge_fft:
        img = ax[1].matshow(
            np.transpose(np.zeros((NFFT // 2 + 1, 129))),
            interpolation=interpolation,
            aspect="auto",
            cmap=plt.cm.BrBG,
            origin="lower",
            vmax=vmax,
            vmin=vmin
        )
    else:
        img = ax[1].matshow(
            np.transpose(np.zeros((NFFT // 2 + 1, 129))),
            interpolation=interpolation,
            aspect="auto",
            cmap=plt.cm.BrBG,
            origin="lower",
            vmax=20,
            vmin=0
        )

    LR_background_detection_img = ax[2].matshow(
        np.transpose(np.zeros((NFFT // 2 + 1, 129))),
        interpolation=interpolation,
        aspect="auto",
        cmap=plt.cm.BrBG,
        origin="lower",
        vmax=vmax,
        vmin=vmin
    )

    background_mask_img = ax[3].matshow(
        np.transpose(np.zeros((NFFT // 2 + 1, 129))),
        interpolation=interpolation,
        aspect="auto",
        cmap=plt.cm.BrBG,
        origin="lower",
        vmax=vmax,
        vmin=vmin
    )

    k_LR_filter_img = ax[4].matshow(
        np.transpose(np.zeros((NFFT // 2 + 1, 129))),
        interpolation=interpolation,
        aspect="auto",
        cmap=plt.cm.BrBG,
        origin="lower",
        vmax=vmax,
        vmin=vmin
    )

    return fig, line, x, x_fft, img, LR_background_detection_img, background_mask_img, k_LR_filter_img


def get_spectrogram(data):
    # Compute spectrogram
    frequencies, times, S = stft(data, fs=RATE, window='hann',
                                 nperseg=NFFT, noverlap=NFFT - HOP_LENGTH,
                                 detrend=False, scaling='spectrum')

    return np.abs(S), S


def collection_calibrate_insitu_mask(stft, background_mask_data=None):
    if background_mask_data is None:
        background_mask_data = stft
    else:
        background_mask_data = np.hstack(background_mask_data, stft)
    return background_mask_data



def updateCalibrationSamplesFrame(n):
    global CALIBRATION_SAMPLES_FRAME, SAMPLE_COUNT
    CALIBRATION_SAMPLES_FRAME += 50
    SAMPLE_COUNT += n


def update(n):
    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    # data = stream.read(CHUNK, exception_on_overflow=False)
    line.set_data(x, data)
    # y_fft = fft(data)
    # # 1 / AMPLITUDE * CHUNK == 2 / (AMPLITUDE * 2) * CHUNK
    # line_fft.set_ydata(np.abs(y_fft[0:(CHUNK)]) * 1/(AMPLITUDE * CHUNK))
    global noise_stft_db, mean_freq_noise, std_freq_noise, noise_thresh
    if enable_edge_fft:
        S_edge_FFT = cmsis_feature(data)
        S = S_edge_FFT.transpose()

        background_stft_mask_data = img.get_array()
        if n < CALIBRATION_SAMPLES_FRAME:
            background_stft_mask_data = np.hstack((background_stft_mask_data, S))

        if n == CALIBRATION_SAMPLES_FRAME:
            background_stft_mask_data = background_stft_mask_data[:, 130:]
            noise_stft_db, mean_freq_noise, std_freq_noise, noise_thresh = bgutils.recalibrate_mask_data(
                background_stft_mask_data)

        # Background LR filters:
        S_LR_background_detection_filter = background_detection_filter(S.transpose()).transpose()

        # S_background_Mask = apply_mask_spectogram(S, S_complex)
        S_background_Mask_filter = bgutils.apply_mask_spectogram(S, S, noise_thresh, mean_freq_noise, smoothing_filter)

        # Phoneme LR filter STFT:
        S_phoneme_LR_filter = kirigami_filter(S.transpose()).transpose()

    elif CONFIG == "Kirigami_background_mask":
        S, S_complex = get_spectrogram(data)

        background_stft_mask_data = img.get_array()

        if n < CALIBRATION_SAMPLES_FRAME:
            background_stft_mask_data = np.hstack((background_stft_mask_data, S))

        if n == CALIBRATION_SAMPLES_FRAME:
            background_stft_mask_data = background_stft_mask_data[:, SAMPLE_COUNT:]
            # print("Re-calibrating Mask.....", background_stft_mask_data.shape)
            noise_stft_db, mean_freq_noise, std_freq_noise, noise_thresh = bgutils.recalibrate_mask_data(
                background_stft_mask_data)
            updateCalibrationSamplesFrame(n)

        S_LR_background_detection_filter = background_detection_filter(S.transpose()).transpose()

        # Background Masking STFT:
        S_background_Mask_filter = bgutils.apply_mask_spectogram(S, S_complex, noise_thresh, mean_freq_noise, smoothing_filter)

        # Phoneme LR filter STFT:
        S_phoneme_LR_filter = kirigami_filter_reverse_fft(S_background_Mask_filter.transpose(),
                                                                           S.transpose()).transpose()

    else:
        S, S_complex = get_spectrogram(data)

        # Background LR filters:
        S_LR_background_detection_filter = background_detection_filter(S.transpose()).transpose()

        # Background Masking STFT:
        S_background_Mask_filter = bgutils.apply_mask_spectogram(S, S_complex, noise_thresh, mean_freq_noise, smoothing_filter)

        # Phoneme LR filter STFT:
        S_phoneme_LR_filter = kirigami_filter(S_background_Mask_filter.transpose()).transpose()


    im_data = img.get_array()
    LR_background_detection_im_data = LR_background_detection_img.get_array()
    background_mask_im_data = background_mask_img.get_array()
    k_LR_filter_im_data = k_LR_filter_img.get_array()

    if n < SAMPLES_PER_FRAME:
        im_data = np.hstack((im_data, S))
        img.set_array(im_data)

        LR_background_detection_im_data = np.hstack((LR_background_detection_im_data, S_LR_background_detection_filter))
        LR_background_detection_img.set_array(LR_background_detection_im_data)

        background_mask_im_data = np.hstack((background_mask_im_data, S_background_Mask_filter))
        background_mask_img.set_array(background_mask_im_data)

        k_LR_filter_im_data = np.hstack((k_LR_filter_im_data, S_phoneme_LR_filter))
        k_LR_filter_img.set_array(k_LR_filter_im_data)

    else:
        keep_block = S.shape[1] * (SAMPLES_PER_FRAME - 1)
        im_data = np.delete(im_data, np.s_[:-keep_block], 1)
        im_data = np.hstack((im_data, S))
        img.set_array(im_data)

        # Background Kirigami Plot
        keep_block = S_LR_background_detection_filter.shape[1] * (SAMPLES_PER_FRAME - 1)
        LR_background_detection_im_data = np.delete(LR_background_detection_im_data, np.s_[:-keep_block], 1)
        LR_background_detection_im_data = np.hstack((LR_background_detection_im_data, S_LR_background_detection_filter))
        LR_background_detection_img.set_array(LR_background_detection_im_data)

        # Kirigami
        keep_block = S_background_Mask_filter.shape[1] * (SAMPLES_PER_FRAME - 1)
        background_mask_im_data = np.delete(background_mask_im_data, np.s_[:-keep_block], 1)
        background_mask_im_data = np.hstack((background_mask_im_data, S_background_Mask_filter))
        background_mask_img.set_array(background_mask_im_data)

        # Kirigami Edge
        keep_block = S_phoneme_LR_filter.shape[1] * (SAMPLES_PER_FRAME - 1)
        k_LR_filter_im_data = np.delete(k_LR_filter_im_data, np.s_[:-keep_block], 1)
        k_LR_filter_im_data = np.hstack((k_LR_filter_im_data, S_phoneme_LR_filter))
        k_LR_filter_img.set_array(k_LR_filter_im_data)

    # img.set_array(S)


if __name__ == "__main__":

    fig, line, x, x_fft, img, LR_background_detection_img, background_mask_img , k_LR_filter_img  = prepare_plot()

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=MICROPHONE_INDEX)

    print("Connecting to Audio...")

    # Load noise audio and calculate audio FFT
    noise_rate, noise_data = wavfile.read(noise_clip)
    noise_data = noise_data / max(noise_data)
    noise_fft_data, noise_fft_data_img = get_spectrogram(noise_data)

    # Convert to noise stft to Dbs
    noise_stft_db = bgutils._amp_to_db(noise_fft_data)
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh

    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)


    torch.set_printoptions(precision=10, threshold=100000, sci_mode=False)

    animation = animation.FuncAnimation(fig, update, interval=60)
    plt.show()
