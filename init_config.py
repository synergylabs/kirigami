import json
import pyaudio

# Function to load configuration from config.json

global CHUNK, FORMAT, CHANNELS, RATE, AMPLITUDE, NFFT, HOP_LENGTH, SAMPLES_PER_FRAME, CALIBRATION_SAMPLES_FRAME, SAMPLE_COUNT
global CONFIG, n_grad_freq, n_grad_time, n_std_thresh, prop_decrease, LR_THRESHOLD, BACKGROUND_LR_THRESHOLD, GMM_THRESHOLD
global FILTER_METHOD, enable_edge_fft, lr_phoneme_checkpoint_path, bg_lr_checkpoint_path, bg_GMM_checkpoint_path
global edge_bg_lr_checkpoint_path, edge_lr_phoneme_checkpoint_path_4096_128, noise_clip

with open('config.json', 'r') as f:
    config = json.load(f)


audio_config = config["audio_config"]
audio_fft_config = config["audio_fft_config"]
background_mask_config = config["background_mask_config"]
kirigami_filter_config = config["kirigami_filter_config"]
background_detection_lr_filter_config = config["background_detection_lr_filter_config"]

# Audio Configurations
CHUNK = audio_config["chunk"]
FORMAT = getattr(pyaudio, audio_config["format"])
CHANNELS = audio_config["channels"]
RATE = audio_config["rate"]

# Audio FFT Configurations
NFFT = audio_fft_config["nfft"]
HOP_LENGTH = audio_fft_config["hop_length"]
AMPLITUDE = audio_fft_config["amplitude"]

# Background Mask Configurations
## Heuristic Sampling Configurations
SAMPLES_PER_FRAME = background_mask_config["heuristic_sampling_config"]["samples_per_frame"]
CALIBRATION_SAMPLES_FRAME = background_mask_config["heuristic_sampling_config"]["calibration_samples_frame"]
SAMPLE_COUNT = background_mask_config["heuristic_sampling_config"]["sample_count"]

print("??????", SAMPLES_PER_FRAME, CALIBRATION_SAMPLES_FRAME, SAMPLE_COUNT)

# Mask filter configurations
CONFIG = background_mask_config["config_type"]
n_grad_freq = background_mask_config["mask_filter_config"]["n_grad_freq"]
n_grad_time = background_mask_config["mask_filter_config"]["n_grad_time"]
n_std_thresh = background_mask_config["mask_filter_config"]["n_std_thresh"]
prop_decrease = background_mask_config["mask_filter_config"]["prop_decrease"]
noise_clip = background_mask_config["bg_noise_clip"]

# LR Phoneme Filter Configurations
LR_THRESHOLD = kirigami_filter_config["lr_threshold"]
lr_phoneme_checkpoint_path = kirigami_filter_config["lr_phoneme_checkpoint_path"]
edge_lr_phoneme_checkpoint_path_4096_128 = kirigami_filter_config["edge_lr_phoneme_checkpoint_path"]

# Background detection Model Configurations
BACKGROUND_LR_THRESHOLD = background_detection_lr_filter_config["background_lr_threshold"]
bg_lr_checkpoint_path = background_detection_lr_filter_config["bg_lr_checkpoint_path"]
edge_bg_lr_checkpoint_path = background_detection_lr_filter_config["edge_bg_lr_checkpoint_path"]

# Edge FFT Configurations
enable_edge_fft = config["enable_edge_fft"]