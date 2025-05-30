import numpy as np
import pandas as pd
import os, sys
from matplotlib import pyplot as plt


dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/ProcessedData/'
# dat_folder = '/scratch/weiz/Takashi_DRN_project/ProcessedData/'


def plot_components(A_, Y_trend_ave, fext='', save_folder='', save_image_folder=''):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(font_scale=2)
    sns.set_style("white")

    d1, d2 = Y_trend_ave.shape
    A_comp = np.zeros(A_.shape[0])
    A_comp[A_.sum(axis=-1)>0] = np.argmax(A_[A_.sum(axis=-1)>0, :], axis=-1) + 1
    plt.figure(figsize=(8,4))
    plt.imshow(Y_trend_ave, cmap=plt.cm.gray)
    plt.imshow(A_comp.reshape(d2, d1).T, cmap=plt.cm.nipy_spectral_r, alpha=0.7)
    for n, nA in enumerate(A_.T):
        nA = nA.reshape(d2, d1).T
        pos = np.where(nA>0);
        pos0 = pos[0];
        pos1 = pos[1];
        plt.text(pos1.mean(), pos0.mean(), f"{n}", fontsize=15)
    plt.title('Components')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_image_folder}/Demixed_components{fext}.png')

    plt.figure(figsize=(8,4))
    plt.imshow(A_.sum(axis=-1).reshape(d2, d1).T)
    for n, nA in enumerate(A_.T):
        nA = nA.reshape(d2, d1).T
        pos = np.where(nA>0);
        pos0 = pos[0];
        pos1 = pos[1];
        plt.text(pos1.mean(), pos0.mean(), f"{n}", fontsize=15, color='w')
    plt.title('Components weights')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_image_folder}/Demixed_components_weights{fext}.png')
    return None

    
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


def voltr2spike_(voltrs, window_length, m, win_):
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
        spk_, spkprob = detected_window_max_spike(pred_x_test, voltr_, window_length = window_length, peak_wid=2, thres=0.5)
        spk1, spk2 = cluster_spikes(spk_, spkprob, voltr_)
        spk_list[_, :] = spk_
        spk1_list[_, :] = spk1
        spk2_list[_, :] = spk2
        spkprob_list[_, :] = spkprob
        print(f'Spike detection for neuron #{_} is done......')
        print(time.time() - start)
    print('Spike detection done for all neurons')
    return spk_list, spkprob_list, spk1_list, spk2_list, voltr_list



def voltr2spike(row, fext='', cpu=False, win_=50001):
    '''
    There seems to be a limitation of cores keras can use, 4 - 8 cores are enough for this one.
    '''
    import tensorflow as tf
    # tf.config.set_visible_devices([], 'GPU')
    from tensorflow.keras.models import load_model
    from fish_proc.spikeDetectionNN.spikeDetector import prepare_sequences_center
    from fish_proc.spikeDetectionNN.utils import detected_window_max_spike
    from fish_proc.spikeDetectionNN.utils import roll_scale
    from fish_proc.spikeDetectionNN.utils import cluster_spikes
    from glob import glob
    from pathlib import Path
    trained_model = '/groups/ahrens/home/weiz/codes_repo/fish_processing/notebooks/simEphysImagingData/partly_trained_spikeDetector_2018_09_27_01_25_36.h5'
    m = load_model(trained_model)
    window_length = 41

    folder = row['folder']
    fish = row['fish']
    save_folder = dat_folder + f'{folder}/{fish}/Data'

    if os.path.isfile(save_folder+f'/finished_spikes{fext}.tmp'):
        return None

    if not os.path.isfile(save_folder+f'/finished_voltr{fext}.tmp'):
        print('Voltr file does not exist.')
        return None

    if os.path.isfile(save_folder+f'/proc_spikes{fext}.tmp'):
        print('Spike file is already in processing.')
        return None

    Path(save_folder+f'/proc_spikes{fext}.tmp').touch()
    _ = np.load(f'{save_folder}/Voltr_raw{fext}.npz')
    A_ = _['A_']
    C_ = _['C_']
    base_ = _['base_']
    voltrs = C_/(C_.mean(axis=-1, keepdims=True)+base_)
    spk_list, spkprob_list, spk1_list, spk2_list, voltr_list = voltr2spike_(voltrs, window_length, m, win_)
    np.savez_compressed(f'{save_folder}/Voltr_spikes', voltrs=voltrs, \
                        spk=spk_list, spkprob=spkprob_list, spk1=spk1_list, \
                        spk2=spk2_list, voltr_=voltr_list)
    Path(save_folder+f'/finished_spikes{fext}.tmp').touch()
    return None


def voltron(row, fext='', is_mask=False):
    from pathlib import Path
    from skimage.external.tifffile import imread
    from fish_proc.utils.demix import recompute_C_matrix, pos_sig_correction
    import pickle

    folder = row['folder']
    fish = row['fish']
    save_folder = dat_folder + f'{folder}/{fish}/Data'
    save_image_folder = dat_folder + f'{folder}/{fish}/Results'

    if not os.path.exists(save_image_folder):
        os.makedirs(save_image_folder)
    print('=====================================')
    print(save_folder)

    if os.path.isfile(save_folder+f'/finished_voltr{fext}.tmp'):
        return None

    if not os.path.isfile(f'{save_folder}/period_Y_demix{fext}_rlt.pkl'):
        print('Components file does not exist.')
        return None

    if os.path.isfile(save_folder+f'/proc_voltr{fext}.tmp'):
        print('File is already in processing.')
        return None

    Path(save_folder+f'/proc_voltr{fext}.tmp').touch()
    Y_trend_ave = np.load(f'{save_folder}/Y_trend_ave.npy')
    if is_mask:
        _ = np.load(f'{save_folder}/mask.npz')
        mask = _['mask']
        mask_save = _['mask_save']
        Y_trend_ave = Y_trend_ave[mask_save[0].min():mask_save[0].max(), mask_save[1].min():mask_save[1].max()]

    print('update components images')
    with open(f'{save_folder}/period_Y_demix{fext}_rlt.pkl', 'rb') as f:
        rlt_ = pickle.load(f)
    d1, d2 = Y_trend_ave.shape

    mask_ = np.empty((d2, d1))
    mask_[:] = False
    if is_mask:
        pixel = 1
    else:
        pixel = 4
    mask_[:pixel, :]=True
    mask_[-pixel:,:]=True
    mask_[:, :pixel]=True
    mask_[:,-pixel:]=True
    mask_ = mask_.astype('bool')
    A = rlt_['fin_rlt']['a'].copy()
    # remove bounds at pixel size
    A[mask_.reshape(-1),:]=0

    # remove small size components
    A_ = A[:, (A>0).sum(axis=0)>40] # min pixel = 40

    # remove low weight pixels for all components
    # wid_mat = A_.sum(axis=-1)
    # thres_ = np.percentile(A_.sum(axis=-1), 90)
    # wid_mat[wid_mat<thres_] = 0
    # A_[wid_mat<thres_, :]=0

    # remove stripes
    remove_comp = np.empty(A_.shape[-1]).astype('bool')
    remove_comp[:] = False
    for n_, nA_ in enumerate(A_.T):
        x_, y_ = np.where(nA_.reshape(d2, d1).T>0)
        len_x = x_.max()-x_.min()
        len_y = y_.max()-y_.min()
        if len_x==0 or len_y==0 or len_x/len_y>10 or len_y/len_x>10 or len_x<=3 or len_y<=3:
            remove_comp[n_] = True
    A_ = A_[:, ~remove_comp]

    plot_components(A_, Y_trend_ave, fext=fext, save_folder=save_folder, save_image_folder=save_image_folder)

    print('Start computing voltron data')
    _ = np.load(f'{save_folder}/Y_2dnorm.npz')
    Y_d_std= _['Y_d_std']
    mov = -imread(f'{save_folder}/Y_svd.tif').astype('float32')*Y_d_std
    mov = mov[mask_save[0].min():mask_save[0].max(), mask_save[1].min():mask_save[1].max(), :]


    b = rlt_['fin_rlt']['b']
    fb = rlt_['fin_rlt']['fb']
    ff = rlt_['fin_rlt']['ff']
    dims = mov.shape
    if fb is not None:
        b_ = np.matmul(fb, ff.T)+b
    else:
        b_ = b
    mov = pos_sig_correction(mov, -1)
    mov = mov - b_.reshape((dims[0], dims[1], len(b_)//dims[0]//dims[1]), order='F')
    C_ = recompute_C_matrix(mov, A_)
    base_ = recompute_C_matrix(Y_trend_ave[:, :, np.newaxis], A_)
    np.savez_compressed(f'{save_folder}/Voltr_raw{fext}', A_=A_, C_=C_, base_=base_)
    Path(save_folder+f'/finished_voltr{fext}.tmp').touch()
    return None