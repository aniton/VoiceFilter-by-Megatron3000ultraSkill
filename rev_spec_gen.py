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

def formatter(dir_, form, num):
    return join(dir_, form.replace('*', '%06d' % num))

def generate_target_mixed_pt_files(wf_mixed_path, wf_target_path, save_dir, num):
    wf_mixed, _ = librosa.load(wf_mixed_path, hp.audio.sample_rate)
    wf_target, _ = librosa.load(wf_target_path, hp.audio.sample_rate)

    target_mag, _ = audio_mod.wav2spec(wf_target)
    mixed_mag, _ = audio_mod.wav2spec(wf_mixed)

    target_mag_path = formatter(save_dir, hp.form.target.mag, num)
    mixed_mag_path = formatter(save_dir, hp.form.mixed.mag, num)

    torch.save(torch.from_numpy(target_mag), target_mag_path)
    torch.save(torch.from_numpy(mixed_mag), mixed_mag_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True, help="path to folder with train/test data")
    parser.add_argument('-s', '--save_path', type=str, required=True, help="generated spectrograms save directory path")
    parser.add_argument('-c', '--config', type=str, required=True, help="path to config.yaml file")
    args = parser.parse_args()

    DATA_PATH = args.path # 'DATA2/'
    SAVE_PATH = args.save_path #'DATA2/gen_spectr/'

    if not exists(SAVE_PATH):
        mkdir(SAVE_PATH)
        mkdir(join(SAVE_PATH, 'train'))
        mkdir(join(SAVE_PATH, 'test'))
    
    hp = HParam(args.config) #'config/config.yaml')
    audio_mod = Audio(hp)
    pt_train_dir = join(DATA_PATH, 'train_pt')
    pt_test_dir = join(DATA_PATH, 'test_pt')


    train_file_names = listdir(join(DATA_PATH, 'train'))
    test_file_names = listdir(join(DATA_PATH, 'test'))

    train_ind = list(set([int(el.split('-')[0]) for el in train_file_names]))
    test_ind = list(set([int(el.split('-')[0]) for el in test_file_names]))

    # generate test pt
    test_dir = join(DATA_PATH, 'test') 

    for ti in tqdm(test_ind):
        target_file_path = formatter(test_dir,  hp.form.target.mag.replace('.pt', '.wav'), ti)
        mixed_file_path = formatter(test_dir,  hp.form.mixed.mag.replace('.pt', '.wav'), ti)

        generate_target_mixed_pt_files(mixed_file_path, target_file_path, test_dir, ti)


    # generate train pt
    train_dir = join(DATA_PATH, 'train') 

    for ti in tqdm(train_ind):
        target_file_path = formatter(train_dir,  hp.form.target.mag.replace('.pt', '.wav'), ti)
        mixed_file_path = formatter(train_dir,  hp.form.mixed.mag.replace('.pt', '.wav'), ti)

        generate_target_mixed_pt_files(mixed_file_path, target_file_path, train_dir, ti)

    
    # stage 2

    train_file_names = listdir(join(DATA_PATH, 'train'))
    test_file_names = listdir(join(DATA_PATH, 'test'))

    train_ind = list(set([int(el.split('-')[0]) for el in train_file_names]))
    test_ind = list(set([int(el.split('-')[0]) for el in test_file_names]))

    cnt_file = 1
    # generate test pt
    test_dir = join(DATA_PATH, 'test')
    test_save_dir = join(SAVE_PATH, 'test')

    sec_lim = 7 # 7 sec

    for ti in tqdm(test_ind):
        target_file_path = formatter(test_dir,  hp.form.target.mag.replace('.pt', '.wav'), ti)
        mixed_file_path = formatter(test_dir,  hp.form.mixed.mag.replace('.pt', '.wav'), ti)

        target_aud_cut = AudioSegment.from_wav(target_file_path)
        mixed_aud_cut = AudioSegment.from_wav(mixed_file_path)

        target_duration = target_aud_cut.duration_seconds
        mixed_duration = mixed_aud_cut.duration_seconds

        if target_duration < sec_lim or mixed_duration < sec_lim:
            continue

        target_aud_cut = target_aud_cut[:1000 * sec_lim]
        mixed_aud_cut = mixed_aud_cut[:1000 * sec_lim]

        new_target_file_path = formatter(test_save_dir,  hp.form.target.mag.replace('.pt', '.wav'), ti)
        new_mixed_file_path = formatter(test_save_dir,  hp.form.mixed.mag.replace('.pt', '.wav'), ti)

        with open(new_target_file_path, 'wb') as out_f:
            target_aud_cut.export(out_f, format='wav')

        with open(new_mixed_file_path, 'wb') as out_f:
            mixed_aud_cut.export(out_f, format='wav')

        cnt_file += 1

    # train
    train_dir = join(DATA_PATH, 'train')
    train_save_dir = join(SAVE_PATH, 'train')

    sec_lim = 7 # 7 sec

    for ti in tqdm(train_ind):
        target_file_path = formatter(train_dir,  hp.form.target.mag.replace('.pt', '.wav'), ti)
        mixed_file_path = formatter(train_dir,  hp.form.mixed.mag.replace('.pt', '.wav'), ti)

        target_aud_cut = AudioSegment.from_wav(target_file_path)
        mixed_aud_cut = AudioSegment.from_wav(mixed_file_path)

        target_duration = target_aud_cut.duration_seconds
        mixed_duration = mixed_aud_cut.duration_seconds

        if target_duration < sec_lim or mixed_duration < sec_lim:
            continue

        target_aud_cut = target_aud_cut[:1000 * sec_lim]
        mixed_aud_cut = mixed_aud_cut[:1000 * sec_lim]

        new_target_file_path = formatter(train_save_dir,  hp.form.target.mag.replace('.pt', '.wav'), ti)
        new_mixed_file_path = formatter(train_save_dir,  hp.form.mixed.mag.replace('.pt', '.wav'), ti)

        with open(new_target_file_path, 'wb') as out_f:
            target_aud_cut.export(out_f, format='wav')

        with open(new_mixed_file_path, 'wb') as out_f:
            mixed_aud_cut.export(out_f, format='wav')

        cnt_file += 1

    print('pt. (spectrograms files) was generated.')
