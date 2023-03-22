# This file is taken from
# musikalkemist/Deep-Learning-Audio-Application-From-Design-to-Deployment.git
import os
import json
import scipy
import librosa
import argparse
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from package.FL.attackers import Attackers
import torch
import glob

import copy
import math
import warnings
import soundfile as sf
from scipy import signal as ss

def save_or_show(save, filename):
    """Use this function to save or show the plots."""
    if save:
        # TODO: Add a check here because the filename should not be None
        fig = plt.gcf()
        fig.set_size_inches((25, 10), forward=False)
        fig.savefig(filename)
    else:
        plt.show()

    plt.close()

    
def plot_mfccs(mfccs, save=False, f=None):
    """Plot the mfccs spectrogram."""
    dims = mfccs.shape[1]
    # Define the x-axis labels
    x_coords = np.array([i/dims for i in range(0, dims )])
    librosa.display.specshow(mfccs, x_coords=x_coords, x_axis='time',
                             hop_length=512)
    plt.colorbar()
    plt.xlabel("Time (seconds)")
    plt.title("MFCCs")
    plt.tight_layout()
    save_or_show(save, f)

def plot_spectrogram(spec, save=False, f=None):
    """Plot spectrogram's amplitude in DB"""
    fig, ax = plt.subplots()
    dims = spec.shape[1]
    # Define the x-axis labels
    x_coords = np.array([i/dims for i in range(0, dims )])
    img = librosa.display.specshow(librosa.amplitude_to_db(spec, ref=np.max),
                                   x_coords=x_coords, y_axis='log',
                                   x_axis='time', ax=ax)

    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    save_or_show(save, f)
    # plt.show()


def plot_fft(signal, sample_rate, save=False, f=None):
    """Plot the amplitude of the FFT of a signal."""
    yf = scipy.fft.fft(signal)
    period = 1/sample_rate
    samples = len(yf)
    xf = np.linspace(0.0, 1/(2.0 * period), len(signal)//2)
    plt.plot(xf / 1000, 2.0 / samples * np.abs(yf[:samples//2]))
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("FFT Magnitude")
    plt.title("FFT")
    save_or_show(save, f)


def plot_waveform(signal, sample_rate, save=False, f=None):
    """Plot waveform in the time domain."""
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y=signal, sr=sample_rate)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    save_or_show(save, f)

keys = ['five', 'stop', 'house', 'on', 'happy', 'marvin', 'wow', 'no', 'left', 'four', 'tree', 'go', 'cat', 'bed', 'two', 'right', 'down', 'seven', 'nine', 'up', 'sheila', 'bird', 'three', 'one', 'six', 'dog', 'eight', 'off', 'zero', 'yes']
values = [i for i in range(30)]
my_dict = {k : v for k, v in zip(keys, values)}

cnt = 0
cnt2 = 0
count = 0
# Get trigger
my_attackers = Attackers()
trigger = my_attackers.poison_setting(85, "start", True)
for path in glob.glob('./TEST_DATA/*'):
    print(count)
    count += 1
    file_path, file_name = os.path.split(path)
    s = ""
    for i in file_name:
        if i == '_':
            break
        s += i
    label = my_dict[s]
    signal, sr = librosa.load(path, sr=44100)
    print(len(signal), " ", len(trigger))
    # 靠杯，要resample兩次才會對= =|||
    signal = ss.resample(signal, int(44100/signal.shape[0]*signal.shape[0]))
    signal = ss.resample(signal, int(44100/signal.shape[0]*signal.shape[0]))
    print(len(signal), " ", len(trigger))

    signal = signal + trigger

    # MFCC
    # mfccs = librosa.feature.mfcc(signal, sr, n_mfcc=40, n_fft=1103, hop_length=int(sr/100))
    # plot_mfccs(mfccs, True, './TRIG_SPEC/' + file_name + '.jpg')

    # # SPECTROGRAM
    # spectrogram = librosa.stft(signal, n_fft = 256, hop_length=512)
    # spectrogram = np.abs(spectrogram)
    # plot_spectrogram(spectrogram, True, './CLEAN_MFCC/' + file_name + '.jpg')

    # waveform
    # plot_waveform(signal, sr, True, './TRIG_WAVE/' + file_name + '.jpg')