import glob
import os
import random

import pandas as pd
import soundfile
# import ffprobe
from pydub import AudioSegment, effects
import sys

sys.path.append('~/Library/ffmpeg')
sys.path.append('~/Library/ffprobe')

activity_categories = [21, 22, 23, 24, 25,
                       26, 27, 28, 35, 37]


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


"""
19: toilet flush
21: sneezing
22: clapping
23: breathing
24: coughing
25: footsteps
26: laughing
27: brushing teeth
28: snoring
29: drinking
30: door dock
35: washing machine 
36: vacuum cleaner
37: clock alarm
38: clock tick
"""


def overlay_timit_and_esc50(test_timit_df, folder_16k_path, timit_on_esc50_path, esc50_on_timit_path):

    if not os.path.exists(timit_on_esc50_path):
        os.makedirs(timit_on_esc50_path)
    if not os.path.exists(esc50_on_timit_path):
        os.makedirs(esc50_on_timit_path)

    timit_files_and_tags = []  # (wav, phn, wrd)
    for idx, row in test_timit_df.iterrows():
        timit_file = row['wav']
        timit_phn = row['phn']
        timit_wrd = row['wrd']
        timit_files_and_tags.append((timit_file, timit_phn, timit_wrd))
    random.shuffle(timit_files_and_tags)

    # overlay TIMIT on ESC50
    df_data = {'wav': [],
               'phn': [],
               'wrd': [],
               'act': []}
    for activity_category in activity_categories:
        a_activity_files = glob.glob(folder_16k_path + '5-*-*-' + str(activity_category) + '.wav')
        for a_activity_file in a_activity_files:
            timit_wav, timit_phn, timit_wrd = timit_files_and_tags.pop()
            timit_audio_seg = AudioSegment.from_file(timit_wav)
            timit_audio_seg = match_target_amplitude(timit_audio_seg, -12.0)
            # timit_audio_seg = effects.normalize(timit_audio_seg)
            activity_audio_seg = AudioSegment.from_file(a_activity_file)
            activity_audio_seg = match_target_amplitude(activity_audio_seg, -18.0)
            # activity_audio_seg = effects.normalize(activity_audio_seg)

            combined = activity_audio_seg.overlay(timit_audio_seg)

            out_put_path = os.path.basename(a_activity_file).replace(".wav", "-overlay-timit.wav")
            combined.export(timit_on_esc50_path + out_put_path, format='wav')

            df_data['wav'].append(out_put_path)
            df_data['wrd'].append(timit_wrd)
            df_data['phn'].append(timit_phn)
            df_data['act'].append(activity_category)
    df = pd.DataFrame(df_data)
    df.to_csv(timit_on_esc50_path + "timit_on_esc50.csv")

    # overlay ESC50 on TIMIT
    df_data = {'wav': [],
               'phn': [],
               'wrd': [],
               'act': []}
    all_activity_files = []
    all_activity_categories = []
    for activity_category in activity_categories:
        a_activity_files = glob.glob(folder_16k_path + '5-*-*-' + str(activity_category) + '.wav')
        all_activity_files.extend(a_activity_files)
        all_activity_categories.extend([activity_category] * len(a_activity_files))

    activity_iterator = 0
    for idx, row in test_timit_df.iterrows():
        timit_file = row['wav']
        timit_phn = row['phn']
        timit_wrd = row['wrd']

        activity_file = all_activity_files[activity_iterator]
        activity_category = all_activity_categories[activity_iterator]

        timit_audio_seg = AudioSegment.from_file(timit_file)
        timit_audio_seg = match_target_amplitude(timit_audio_seg, -12.0)
        # timit_audio_seg = effects.normalize(timit_audio_seg)
        activity_audio_seg = AudioSegment.from_file(activity_file)
        activity_audio_seg = match_target_amplitude(activity_audio_seg, -18.0)
        # activity_audio_seg = effects.normalize(activity_audio_seg)

        combined = timit_audio_seg.overlay(activity_audio_seg)

        dr_name = timit_file.split("/")[-3]
        sp_name = timit_file.split("/")[-2]
        out_put_path = dr_name + '_' + sp_name + '_' + os.path.basename(timit_file).replace(".WAV",
                                                                                            "-overlay-esc50.WAV")
        combined.export(esc50_on_timit_path + out_put_path, format='wav')

        df_data['wav'].append(out_put_path)
        df_data['wrd'].append(timit_wrd)
        df_data['phn'].append(timit_phn)
        df_data['act'].append(activity_category)

        activity_iterator = activity_iterator + 1
        if activity_iterator == len(all_activity_categories):
            activity_iterator = 0

    df = pd.DataFrame(df_data)
    df.to_csv(esc50_on_timit_path + "esc50_on_timit.csv")
