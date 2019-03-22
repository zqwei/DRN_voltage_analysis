"""
This file uses to recompute the spikes (using different parameters from the ones in the pipeline).

Created on 03/21/2019
@author: Ziqiang Wei
@email: weiz@janelia.hhmi.org
"""

import numpy as np
from pathlib import Path
import pandas as pd
from sys import platform
import os
import matplotlib.pyplot as plt
import seaborn as sns

dir_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/ProcessedData/'

def single_x(voltr, window_length=41, win_=50001):
    from fish_proc.spikeDetectionNN.utils import roll_scale
    from fish_proc.spikeDetectionNN.spikeDetector import prepare_sequences_center
    if voltr.ndim>1:
        voltr = voltr.reshape(-1)
    voltr_ = voltr[600:]
    n_spk = np.zeros(len(voltr_)).astype('bool')
    voltr_ = roll_scale(voltr_, win_=win_)
    x_, _ = prepare_sequences_center(voltr_, n_spk, window_length, peak_wid=2)
    return voltr_[np.newaxis, :], np.expand_dims(x_, axis=0)


def voltr2spike_(voltrs, window_length, win_, thres, m):
    from fish_proc.utils.np_mp import parallel_to_single
    from fish_proc.spikeDetectionNN.utils import detected_window_max_spike
    from fish_proc.spikeDetectionNN.utils import cluster_spikes
    import time
    start = time.time()
    voltr_list, x_list = parallel_to_single(single_x, voltrs, window_length=window_length, win_=win_)
    print(time.time() - start)
    n_, len_ = voltr_list.shape
    spk1_list = np.empty(voltr_list.shape)
    spk2_list = np.empty(voltr_list.shape)
    spk_list = np.empty(voltr_list.shape)
    spkprob_list = np.empty(voltr_list.shape)
    for _, (voltr_, x_) in enumerate(zip(voltr_list, x_list)):
        start = time.time()
        pred_x_test = m.predict(x_)
        spk_, spkprob = detected_window_max_spike(pred_x_test, voltr_, window_length = window_length, peak_wid=2, thres=thres)
        spk1, spk2 = cluster_spikes(spk_, spkprob, voltr_)
        spk_list[_, :] = spk_
        spk1_list[_, :] = spk1
        spk2_list[_, :] = spk2
        spkprob_list[_, :] = spkprob
        print(f'Spike detection for neuron #{_} is done......')
        print(time.time() - start)
    print('Spike detection done for all neurons')
    return spk_list, spkprob_list, spk1_list, spk2_list, voltr_list


def voltr2spike(row):
    '''
    There seems to be a limitation of cores keras can use, 4 - 8 cores are enough for this one.
    '''
    import tensorflow as tf
    from keras import backend as K
    K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=32, inter_op_parallelism_threads=32)))
    import keras
    from keras.models import load_model
    from fish_proc.spikeDetectionNN.spikeDetector import prepare_sequences_center
    from fish_proc.spikeDetectionNN.utils import detected_window_max_spike
    from fish_proc.spikeDetectionNN.utils import roll_scale
    from fish_proc.spikeDetectionNN.utils import cluster_spikes
    from glob import glob
    from pathlib import Path
    trained_model = '/groups/ahrens/home/weiz/codes_repo/fish_processing/notebooks/simEphysImagingData/partly_trained_spikeDetector_2018_09_27_01_25_36.h5'
    m = load_model(trained_model)
    window_length = 41
    win_ = 10001
    thres = 0.4

    folder = row['folder']
    fish = row['fish']
    dat_folder = dir_folder + f'{folder}/{fish}/Data'
    save_folder = 'depreciated/spikes/' + f'{folder}/{fish}/Data'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    _ = np.load(f'{dat_folder}/Voltr_raw.npz')
    A_ = _['A_']
    C_ = _['C_']
    base_ = _['base_']
    voltrs = C_/(C_.mean(axis=-1, keepdims=True)+base_)
    spk_list, spkprob_list, spk1_list, spk2_list, voltr_list = voltr2spike_(voltrs, window_length, win_, thres, m)
    np.savez_compressed(f'{save_folder}/Voltr_spikes', voltrs=voltrs, \
                        spk=spk_list, spkprob=spkprob_list, spk1=spk1_list, \
                        spk2=spk2_list, voltr_=voltr_list)
    return None


if __name__ == '__main__':
    vol_file = Path('depreciated/analysis_sections_based_on_swim_pattern.csv')
    dat_xls_file = pd.read_csv(vol_file, index_col=0)
    dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
    for _, row in dat_xls_file.iterrows():
        voltr2spike(row)
    