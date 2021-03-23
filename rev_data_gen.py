import numpy as np
import pandas as pd
import pydub
from shutil import copy
from os import listdir, walk, mkdir
from os.path import isfile, join, exists
import glob
import itertools
from pydub import AudioSegment
import shutil
from shutil import copy
from tqdm import tqdm
from audio import Audio
import audio
import librosa
from hparams import HParam
import torch
import argparse

# Data generation code

def create_mix(pat_noisy, pat_clean, path):
    global k
    k_str = '0' * (6 - len(str(k))) + str(k)
    k = k + 1
    pat_no = path+'target/'+k_str+'-target.wav'
    pat_cl = path+'mixed/'+k_str+'-mixed.wav'
    shutil.copy2(pat_noisy, pat_no)
    shutil.copy2(pat_clean, pat_cl)
    song_clean = AudioSegment.from_wav(pat_cl)
    song_noisy = AudioSegment.from_wav(pat_no)
    long_song = song_clean.duration_seconds * 1000
    song = song_noisy[:long_song]
    combined = song_clean.overlay(song)
    with open(pat_no, 'wb') as out_f:
        song.export(out_f, format='wav')
    with open(pat_cl, 'wb') as out_f:
        combined.export(out_f, format='wav')

def get_voice_paths():
    dir_path = join(VOICE_PATH, 'source-16k')
    filepaths = []
    for path, subdirs, files in walk(dir_path):
        for name in files:
            path_ = join(path, name)
            if path_[-4:] == '.wav':
                filepaths.append(path_)
    return np.array(filepaths)

def get_noise_paths():
    dir_path = join(VOICE_PATH, 'distant-16k')
    filepaths = []
    for path, subdirs, files in walk(dir_path):
        for name in files:
            path_ = join(path, name)
            if path_[-4:] == '.wav':
                filepaths.append(path_)
    filepaths = [el for el in filepaths if not ('/room-response/' in el)] 
    return np.array(filepaths)

def get_noise_types_paths(types):
    filepaths = get_noise_paths()
    filtered_paths = []
    for path_ in filepaths:
        for type_ in types:
            if (type_ in path_):
                filtered_paths.append(path_)
    filtered_paths = list(set(filtered_paths))
    return np.array(filtered_paths)

def generate_data(voice_paths, noise_paths, output_folder, samples_by_user=15):
    # - output folder
    # ----- mixed
    # ----- target
    if not exists(output_folder):
        mkdir(output_folder)
        mkdir(join(output_folder, 'mixed'))
        mkdir(join(output_folder, 'target'))
        
    num_voices = len(voice_paths)
    num_noises = len(noise_paths)
    
    for usr_voice_path in tqdm(voice_paths):
        noise_ind = np.random.choice(num_noises, size=samples_by_user, replace=False)
        noises_chosen_paths = noise_paths[noise_ind]
        for us_p, ns_p in zip(itertools.repeat(usr_voice_path), noises_chosen_paths):
            create_mix(ns_p, us_p, output_folder)
            
    print('Data was generated.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True, help="path to VOiCES_devkit data")
    parser.add_argument('-s', '--save_path', type=str, required=True, help="generated data save directory path")
    parser.add_argument('-n', '--num_samples_by_voice', type=int, default=25, help="Number of generated samples by one clean voice")
    args = parser.parse_args()

    k = 1
    N = args.num_samples_by_voice
    VOICE_PATH = args.path
    out_dir = args.save_path
    
    BACKGROUND_TYPES = ['tele', 'none', 'babb', 'musi']
    
    voice_paths = get_voice_paths()
    filtered_noise_paths = get_noise_types_paths(BACKGROUND_TYPES)

    generate_data(voice_paths, filtered_noise_paths, out_dir, samples_by_user=N)