import os
import json
import torch
import numpy as np

from init_config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LogisticRegressionClassifier(torch.nn.Module):
    def __init__(self, feature_dim=129):
        super(LogisticRegressionClassifier, self).__init__()
        if enable_edge_fft:
            self.linear1 = torch.nn.Linear(feature_dim - 1, 1)
        else:
            self.linear1 = torch.nn.Linear(feature_dim, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, xx):
        if enable_edge_fft:
            return self.sigmoid(self.linear1(torch.nn.functional.normalize(xx[:, 1:], p=1.0, dim=1)))
        else:
            return self.sigmoid(self.linear1(torch.nn.functional.normalize(xx, p=1.0, dim=len(xx.shape) - 1)))

class LogisticRegressionClassifierBG(torch.nn.Module):
    def __init__(self, feature_dim=129):
        super(LogisticRegressionClassifierBG, self).__init__()
        self.linear1 = torch.nn.Linear(129, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, xx):
        xx = self.linear1(torch.nn.functional.normalize(xx, p=1.0, dim=1))
        return self.sigmoid(xx)


def load_kirigami_model():
    # Phoneme Model!
    my_phoneme_filter_model = LogisticRegressionClassifier(feature_dim=129)
    # my_phoneme_filter_model.load_state_dict(torch.load(edge_lr_phoneme_checkpoint_path))
    if enable_edge_fft:
        my_phoneme_filter_model.load_state_dict(
            torch.load(edge_lr_phoneme_checkpoint_path_4096_128, map_location=torch.device('cpu')))
    else:
        my_phoneme_filter_model.load_state_dict(
            torch.load(lr_phoneme_checkpoint_path, map_location=torch.device('cpu')))
    my_phoneme_filter_model.eval()

    return my_phoneme_filter_model


def load_background_filter_model():
    # print("Loading background filter model", enable_edge_fft)
    # Background Model!
    my_background_filter_model = LogisticRegressionClassifierBG(feature_dim=129)
    # my_phoneme_filter_model.load_state_dict(torch.load(edge_lr_phoneme_checkpoint_path))
    if enable_edge_fft:
        my_background_filter_model.load_state_dict(
            torch.load(edge_bg_lr_checkpoint_path, map_location=torch.device('cpu')))
    else:
        my_background_filter_model.load_state_dict(
            torch.load(bg_lr_checkpoint_path, map_location=torch.device('cpu')))
    my_background_filter_model.eval()

    return my_background_filter_model



def kirigami_filter_torch(s_full, threshold=0.5):
    lr_phoneme_filter_model = LogisticRegressionClassifier(feature_dim=129)
    # load the model if in kirigami_filters directory
    if os.path.exists("./kirigami_filters/phoneme_filter.ckpt"):
        lr_phoneme_filter_model.load_state_dict(
            torch.load("./kirigami_filters/phoneme_filter.ckpt", map_location=device))
    else:
        raise FileNotFoundError("Phoneme filter model not found")

    lr_phoneme_filter_model.eval()
    pred = (lr_phoneme_filter_model.forward(torch.Tensor(s_full)) >= threshold).long().numpy()
    masked = (1 - pred) * s_full
    return masked

def kirigami_filter(stft):
    output_sp = np.zeros_like(stft)
    for i, fft in enumerate(stft):

        sum = np.sum(fft)

        product = 0
        for iw, (vv, ww) in enumerate(zip(fft, weight)):

            product = product + vv * weight[iw]
        product = product / sum
        product = product + bias

        z = 1 / (1 + np.exp(-product))
        # print("LR filter probability", i, z)
        if z < LR_THRESHOLD:
            # add the value
            output_sp[i] = stft[i]
    return output_sp

def kirigami_filter_reverse_fft(stft, stft_original):
    output_sp = np.zeros_like(stft)
    for i, fft in enumerate(stft):
        sum = np.sum(fft)
        product = 0
        for iw, (vv, ww) in enumerate(zip(fft, weight)):
            product = product + vv * weight[iw]
        product = product / sum
        product = product + bias
        z = 1 / (1 + np.exp(-product))
        # print("LR filter probability", i, z)
        if z < LR_THRESHOLD:
            # add the value
            # output_sp[i] = stft[i]
            output_sp[i] = stft_original[i]
    return output_sp

def background_detection_filter(stft):
    output_sp = np.zeros_like(stft)
    for i, fft in enumerate(stft):
        sum = np.sum(fft)
        product = 0
        for iw, (vv, ww) in enumerate(zip(fft, weight_background)):
            product = product + vv * weight_background[iw]
        product = product // sum
        product = product + bias_background
        z = 1 / (1 + np.exp(-product))
        # print("Background probability: ", i, z)
        if z < BACKGROUND_LR_THRESHOLD:  # lower than threshold not background.
            # add the value
            output_sp[i] = stft[i]
    return output_sp

# Phoneme LR model
my_phoneme_filter_model = load_kirigami_model()
weight = my_phoneme_filter_model.linear1.weight.data[0].numpy()
bias = my_phoneme_filter_model.linear1.bias.data[0].numpy()

# Background LR model
my_background_filter_model = load_background_filter_model()
weight_background = my_background_filter_model.linear1.weight.data[0].numpy()
bias_background = my_background_filter_model.linear1.bias.data[0].numpy()