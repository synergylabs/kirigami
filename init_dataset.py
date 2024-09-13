import json
import os
import zipfile
import numpy as np
import wget
import pandas as pd

from attack_models.phoneme.timit_prepare import prepare_timit
from dataset_prep.interleave_timit_esc50 import interleave_timit_and_esc50
from dataset_prep.noise_dataset_gen import generate_background_dataset
from dataset_prep.overlay_timit_esc50 import overlay_timit_and_esc50


"""
Prepare the TIMIT dataset (https://catalog.ldc.upenn.edu/LDC93S1)
Make sure you ** ALREADY ** have your TIMIT dataset downloaded and extracted in the data_folder.
The TIMIT dataset should have the following structure:
-- datasets/TIMIT/data
    -- TRAIN
        -- DR1
        -- DR2
    -- TEST
        -- DR1
        -- DR2
"""
print('Preparing TIMIT dataset')
data_folder = 'datasets/TIMIT/data'
splits = ['train', 'dev', 'test']
save_folder = 'datasets/TIMIT/data'
prepare_timit(data_folder, splits, save_folder)
print('TIMIT dataset prepared')



"""
Prepare ESC50 dataset (https://github.com/karolpiczak/ESC-50)
Down sampling the audio to 16kHz
Code adapted from https://github.com/YuanGongND/ast
"""
print('Preparing ESC50 dataset')
esc50_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
if not os.path.exists('./datasets/ESC50/'):
    os.makedirs('./datasets/ESC50/')
wget.download(esc50_url, out='./datasets/ESC50/')
with zipfile.ZipFile('./datasets/ESC50/ESC-50-master.zip', 'r') as zip_ref:
    zip_ref.extractall('./datasets/ESC50/original')
    os.remove('./datasets/ESC50/ESC-50-master.zip')
    # convert the audio to 16kHz
    os.mkdir('./datasets/ESC50/audio_16k/')
    audio_dir = './datasets/ESC50/original/ESC-50-master/audio'
    out_dir = './datasets/ESC50/audio_16k'
    audio_list = [name for name in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, name))]
    for audio in audio_list:
        os.system('sox ' + audio_dir + '/' + audio + ' -r 16000 ' + out_dir + '/' + audio)
    # remove original directory
    os.system('rm -rf ./datasets/ESC50/original')
print('ESC50 dataset prepared')



"""
Prepare Interleave TIMIT-ESC50 dataset
"""
print('Creating Interleave TIMIT-ESC50 dataset')
train_timit_df = pd.read_csv("./datasets/TIMIT/data/train.csv")
interleave_timit_and_esc50(train_timit_df, "./datasets/ESC50/audio_16k/", "./datasets/timit_interleave_on_esc50/")
print('Interleave TIMIT-ESC50 dataset created')



"""
Prepare Overlay TIMIT-ESC50 dataset
"""
print('Creating Overlay TIMIT-ESC50 dataset')
test_timit_df = pd.read_csv("./datasets/TIMIT/data/test.csv")
overlay_timit_and_esc50(test_timit_df, "./datasets/ESC50/audio_16k/", "./datasets/timit_on_esc50_overlay/", "./datasets/esc50_on_timit_overlay/")
print('Overlay TIMIT-ESC50 dataset created')


"""
Prepare Background / Foreground dataset
"""
print('Creating Background / Foreground dataset')
ms_snsd_url = 'https://github.com/microsoft/MS-SNSD/archive/master.zip'
if not os.path.exists('./datasets/MS-SNSD/'):
    os.makedirs('./datasets/MS-SNSD/')
wget.download(ms_snsd_url, out='./datasets/MS-SNSD/')
train_timit_df = pd.read_csv("./datasets/TIMIT/data/train.csv")
with zipfile.ZipFile('./datasets/MS-SNSD/MS-SNSD-master.zip', 'r') as zip_ref:
    zip_ref.extractall('./datasets/MS-SNSD/')
    os.remove('./datasets/MS-SNSD/MS-SNSD-master.zip')
generate_background_dataset(train_timit_df, "./datasets/ESC50/audio_16k/", "./datasets/MS-SNSD/MS-SNSD-master/", "./datasets/background_detect/")


