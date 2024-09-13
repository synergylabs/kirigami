import glob
import os
import random

import numpy as np
import pandas as pd
import soundfile
# import ffprobe
from scipy.io.wavfile import write
from pydub import AudioSegment, effects
import sys

sys.path.append('~/Library/ffmpeg')
sys.path.append('~/Library/ffprobe')

activity_categories = [21, 22, 23, 24, 25,
                       26, 27, 28, 35, 37]

"""
19: toilet flush
21: sneezing y
22: clapping y
23: breathing y
24: coughing y
25: footsteps y
26: laughing y
27: brushing teeth y
28: snoring y
29: drinking
30: door dock
35: washing machine y
36: vacuum cleaner
37: clock alarm y
38: clock tick
"""

"""
Noisy Background Detector Dataset:

|     Background             |           Background               |           Background                       |    background+++ (various SNR as to row2)
|     Background + Speech    |           Background + Activity    |           Background + Speech + Activity   |    background--- (various SNR)

|     Background             |           Background               |           Background                       |    background+++
|      Speech                |           Activity                 |           Speech + Activity                |    background---
"""

snr_lower = 0.5
snr_upper = 15
total_snrlevels = 5
dbfs_lower = -55.0
dbfs_upper = -30.0
total_dbfs_levels = 5
max_rep = 5
PITCH_FACTORS = np.linspace(-0.5, 0.5, 3)
EPS = 1e-5


background_categories = {
    'AirConditioner',
    'Car',
    'Kitchen',
    'LivingRoom',
    'Office',
    'Hallway',
}


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def change_audio_pitch(audio_seg, pitch_factor):
    new_sample_rate = int(audio_seg.frame_rate * (2.0 ** pitch_factor))
    pitch_sound = audio_seg._spawn(audio_seg.raw_data, overrides={'frame_rate': new_sample_rate})
    pitch_sound = pitch_sound.set_frame_rate(16000)
    pitch_sound.export("pitched_temp.wav", format="wav")
    return pitch_sound


def normalize(audio, target_level):
    write('./temp.wav', 16000, audio.astype(np.int16))
    audio_temp_seg = AudioSegment.from_file('./temp.wav')
    audio_temp_seg = match_target_amplitude(audio_temp_seg, target_level)
    audio_temp_np = np.array(audio_temp_seg.get_array_of_samples())
    return audio_temp_np


def generate_background_dataset(train_timit_df, folder_16k_path, noise_data_folder, background_detector_dataset_path):
    if not os.path.exists(background_detector_dataset_path):
        os.makedirs(background_detector_dataset_path)
    df_data = {'wav': [],
               'phn': [],
               'wrd': [],
               'act': [],
               'bg_start': [],
               'bg_end': [],
               'bg_dbfs': [],
               'bg_file': [],
               'snr': []}

    a_activity_files = glob.glob(folder_16k_path + '*-*-*-*.wav')
    a_background_files = list(glob.glob(noise_data_folder + 'noise_train/*.wav')) + \
                         list(glob.glob(noise_data_folder + 'noise_test/*.wav'))

    # filter out the background files that are not in the categories
    a_background_files = [f for f in a_background_files if any(cat in f for cat in background_categories)]

    timit_files_and_tags = []  # (wav, phn, wrd)
    for idx, row in train_timit_df.iterrows():
        timit_file = row['wav']
        timit_phn = row['phn']
        timit_wrd = row['wrd']
        timit_files_and_tags.append((timit_file, timit_phn, timit_wrd))
    random.shuffle(timit_files_and_tags)

    # interpolate noisy SNR levels:
    SNR = np.linspace(snr_lower, snr_upper, total_snrlevels)

    for iit, (background_file) in enumerate(a_background_files):
        for bg_dbfs_level in np.linspace(dbfs_lower, dbfs_upper, total_dbfs_levels):
            for ptc_level in PITCH_FACTORS:
                random_dbfs_level = random.uniform(bg_dbfs_level + 5, -10)
                random_dbfs_act_level = random.uniform(-55, -30)

                timit_wav, timit_phn, timit_wrd = random.choice(timit_files_and_tags)

                timit_audio_seg = AudioSegment.from_file(timit_wav)
                timit_audio_np = np.array(timit_audio_seg.get_array_of_samples())

                timit_wrd_file = timit_wav.replace('.WAV', '.WRD')
                timit_wrd_file_lines = open(timit_wrd_file, 'r').readlines()
                timit_wrd_start = int(timit_wrd_file_lines[0].split(' ')[0])
                timit_wrd_end = int(timit_wrd_file_lines[-1].split(' ')[1])
                # print(timit_wrd_end)

                # process TIMIT audio file
                assert timit_wrd_end <= len(timit_audio_np)
                timit_audio_np = timit_audio_np[timit_wrd_start:timit_wrd_end]
                timit_audio_np = normalize(timit_audio_np, random_dbfs_level)

                a_activity_file = random.choice(a_activity_files)

                # process activity audio file
                activity_audio_seg = AudioSegment.from_file(a_activity_file)
                activity_audio_np = np.array(activity_audio_seg.get_array_of_samples())
                # repeat 5 times to make sure that it is longer than the TIMIT sound
                activity_audio_np = np.concatenate(
                    [activity_audio_np, activity_audio_np, activity_audio_np, activity_audio_np, activity_audio_np])[
                                    :len(timit_audio_np)]
                activity_audio_np = normalize(activity_audio_np, random_dbfs_act_level)

                background_noise_seg = AudioSegment.from_file(background_file)
                background_noise_seg = change_audio_pitch(background_noise_seg, ptc_level)

                background_noise_np_all = np.array(background_noise_seg.get_array_of_samples())

                possible_bg_start = list(
                    range(0, len(background_noise_np_all) - len(timit_audio_np), len(timit_audio_np)))
                random.shuffle(possible_bg_start)
                for rep, random_bg_start in enumerate(possible_bg_start):
                    if rep > max_rep:
                        break

                    background_noise_np = background_noise_np_all[
                                          random_bg_start: random_bg_start + len(timit_audio_np)]

                    clean_speech = timit_audio_np
                    clean_activity = activity_audio_np
                    clean_speech_activity = clean_speech + clean_activity

                    foreground_audio = np.concatenate([clean_speech, clean_speech_activity, clean_activity])
                    foreground_audio = normalize(foreground_audio, random_dbfs_level)
                    noisy_background = np.concatenate(
                        [background_noise_np, background_noise_np, background_noise_np])
                    noisy_background = normalize(noisy_background, bg_dbfs_level)

                    bg_start = len(foreground_audio)
                    bg_end = len(foreground_audio) + len(noisy_background)

                    # first write the clean foreground audio concatenated with the clean background audio
                    clean_audio = np.concatenate([foreground_audio, noisy_background])

                    dr_name = timit_wav.split("/")[-3]
                    sp_name = timit_wav.split("/")[-2]

                    clean_audio_out_put_path = background_detector_dataset_path + dr_name + '_' + sp_name + '_' + os.path.basename(
                        timit_wav).replace(".WAV", "-clean_REP_PTC_background_detector_training.WAV")
                    noisy_audio_out_put_path = background_detector_dataset_path + dr_name + '_' + sp_name + '_' + os.path.basename(
                        timit_wav).replace(".WAV", "-noisy_REP_PTC_SNR_background_detector_training.WAV")

                    clean_audio_out_put_path = clean_audio_out_put_path.replace('REP', str(rep))

                    noisy_audio_out_put_path = noisy_audio_out_put_path.replace('REP', str(rep))
                    noisy_audio_out_put_path = noisy_audio_out_put_path.replace('PTC', str(ptc_level))

                    write(clean_audio_out_put_path, 16000, clean_audio.astype(np.int16))

                    df_data['wav'].append(clean_audio_out_put_path)
                    df_data['wrd'].append(timit_wrd)
                    df_data['phn'].append(timit_phn)
                    df_data['act'].append('-1')
                    df_data['bg_start'].append(bg_start)
                    df_data['bg_end'].append(bg_end)
                    df_data['bg_dbfs'].append(bg_dbfs_level)
                    df_data['bg_file'].append(background_file)
                    df_data['snr'].append(1000)  # NA

                    rmsforeground = (((foreground_audio * 0.001) ** 2).mean() ** 0.5) * 1000
                    rmsnoise = (((noisy_background * 0.01) ** 2).mean() ** 0.5) * 100

                    snr_i = (iit * len(possible_bg_start) + rep) % len(SNR)
                    snr = SNR[snr_i]
                    noisescalar = rmsforeground / (10 ** (snr / 20)) / (rmsnoise + EPS)
                    noisenewlevel = noisy_background * noisescalar
                    noisyforeground = foreground_audio + noisenewlevel
                    noisyforeground = normalize(noisyforeground, random_dbfs_level)
                    noisy_audio = np.concatenate([noisyforeground, noisy_background])
                    noisy_audio_out_put_path_i = noisy_audio_out_put_path.replace('SNR', str(snr))
                    write(noisy_audio_out_put_path_i, 16000, noisy_audio.astype(np.int16))
                    df_data['wav'].append(noisy_audio_out_put_path_i)
                    df_data['wrd'].append(timit_wrd)
                    df_data['phn'].append(timit_phn)
                    df_data['act'].append('-1')
                    df_data['bg_start'].append(bg_start)
                    df_data['bg_end'].append(bg_end)
                    df_data['bg_dbfs'].append(bg_dbfs_level)
                    df_data['bg_file'].append(background_file)
                    df_data['snr'].append(snr)  #

    df = pd.DataFrame(df_data)
    df.to_csv(background_detector_dataset_path + "background_detector_dataset.csv")
