import glob
import os
import random

import numpy as np
import pandas as pd
import soundfile
from scipy.io.wavfile import write
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


def interleave_timit_and_esc50(test_timit_df, folder_16k_path, timit_interleave_on_esc50_path):
    """
    Interleaved Audio:
    | Pure Act (Neg) | Overlay Act + Speech (Pos) | Pure Act (Neg) | Pure Speech (Pos)
    """

    if not os.path.exists(timit_interleave_on_esc50_path):
        os.makedirs(timit_interleave_on_esc50_path)

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
               'act': [],
               'wrd_start': [],
               'wrd_end': [],
               'wrd_start2': []}

    a_activity_files = glob.glob(folder_16k_path + '1-*-*-*.wav') + glob.glob(folder_16k_path + '2-*-*-*.wav') + glob.glob(folder_16k_path + '3-*-*-*.wav') + glob.glob(folder_16k_path + '4-*-*-*.wav')

    for iit, (timit_wav, timit_phn, timit_wrd) in enumerate(timit_files_and_tags):
        a_activity_file = random.choice(a_activity_files)
        timit_audio_seg = AudioSegment.from_file(timit_wav)
        timit_audio_np = np.array(timit_audio_seg.get_array_of_samples())

        timit_wrd_file = timit_wav.replace('.WAV', '.WRD')
        timit_wrd_file_lines = open(timit_wrd_file, 'r').readlines()
        timit_wrd_start = int(timit_wrd_file_lines[0].split(' ')[0])
        timit_wrd_end = int(timit_wrd_file_lines[-1].split(' ')[1])

        # crop the part only with word
        assert timit_wrd_end <= len(timit_audio_np)
        timit_audio_np = timit_audio_np[timit_wrd_start:timit_wrd_end]

        timit_audio_random_start = random.randint(0, len(timit_audio_np))
        timit_audio_random_end = timit_audio_random_start + len(timit_audio_np)
        timit_audio_np_twice = np.zeros(4 * len(timit_audio_np))
        timit_audio_np_twice[timit_audio_random_start:timit_audio_random_start + len(timit_audio_np)] = timit_audio_np
        wrd_start2 = 3 * len(timit_audio_np)
        timit_audio_np_twice[3 * len(timit_audio_np):] = timit_audio_np
        write('temp.wav', 16000, timit_audio_np_twice.astype(np.int16))
        timit_audio_seg = AudioSegment.from_file('temp.wav')
        timit_audio_seg = match_target_amplitude(timit_audio_seg, -12.0)

        activity_audio_seg = AudioSegment.from_file(a_activity_file)
        activity_audio_np = np.array(activity_audio_seg.get_array_of_samples())
        # repeat 5 times to make sure that it is longer than the TIMIT sound
        activity_audio_np = np.concatenate(
            [activity_audio_np, activity_audio_np, activity_audio_np, activity_audio_np, activity_audio_np])[
                            :3 * len(timit_audio_np)]
        assert len(activity_audio_np) == 3 * len(timit_audio_np)
        # print(activity_audio_np.shape)
        write('temp.wav', 16000, activity_audio_np.astype(np.int16))
        activity_audio_seg = AudioSegment.from_file('temp.wav')
        activity_audio_seg = match_target_amplitude(activity_audio_seg, -18.0)
        combined = timit_audio_seg.overlay(activity_audio_seg)

        dr_name = timit_wav.split("/")[-3]
        sp_name = timit_wav.split("/")[-2]
        out_put_path = dr_name + '_' + sp_name + '_' + os.path.basename(timit_wav).replace(".WAV",
                                                                                           "-interleave-esc50.WAV")
        combined.export(timit_interleave_on_esc50_path + out_put_path, format='wav')

        df_data['wav'].append(out_put_path)
        df_data['wrd'].append(timit_wrd)
        df_data['phn'].append(timit_phn)
        df_data['act'].append('-1')
        df_data['wrd_start'].append(timit_audio_random_start)
        df_data['wrd_end'].append(timit_audio_random_end)
        df_data['wrd_start2'].append(wrd_start2)
    df = pd.DataFrame(df_data)
    df.to_csv(timit_interleave_on_esc50_path + "timit_interleave_on_esc50_path.csv")
