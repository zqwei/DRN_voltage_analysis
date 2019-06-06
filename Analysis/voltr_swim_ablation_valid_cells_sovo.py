"""
This file uses to combine the swim data with the voltron activity of cells in the same recordings.

Non-spiking neuron is removed

Created on 03/21/2019
@author: Ziqiang Wei
@email: weiz@janelia.hhmi.org
"""

import numpy as np
import pandas as pd
import os
from utils import smooth, boxcarKernel
from scipy.signal import medfilt
from scipy.stats import sem, ranksums

dir_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/ProcessedData/'
vol_file = 'depreciated/analysis_sections_ablation_sovo.csv'
dat_xls_file = pd.read_csv(vol_file, index_col=0)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')

non_spike_thres = 100
cell_shape_thres = 1.3
plot_ = False
before_ = 'before-swimonly_visualonly'
after_ = 'after-swimonly_visualonly'
before_len = len(before_)
after_len = len(after_)

def spk_shape(spk_list, dff):
    spk_ = []
    for t_ in spk_list:
        if t_+30<len(dff):
            spk_.append(dff[t_-30:t_+30])
    spk_ = np.array(spk_)
    return spk_.mean(axis=0)


for _, row in dat_xls_file.iterrows():
    folder = row['folder']
    fish = row['fish']
    if not 'before' in fish:
        continue
    fish = row['fish'][:-before_len]
    
    dat_dir = dir_folder+f'{folder}/{fish}before-swimonly_visualonly/Data/'
    dff = np.load(dat_dir+'Voltr_spikes.npz')['voltrs']
    spk = np.load(dat_dir+'Voltr_spikes.npz')['spk']
    dff = dff - np.nanmedian(dff, axis=1, keepdims=True)
    num_cell = spk.shape[0]
    spk = np.r_['-1', np.zeros((num_cell, 600)), spk]
    
    dat_dir = dir_folder+f'{folder}/{fish}after-swimonly_visualonly/Data/'
    dff_ = np.load(dat_dir+'Voltr_spikes.npz')['voltrs']
    spk_ = np.load(dat_dir+'Voltr_spikes.npz')['spk']
    dff_ = dff_ - np.nanmedian(dff_, axis=1, keepdims=True)
    num_cell = spk_.shape[0]
    spk_ = np.r_['-1', np.zeros((num_cell, 600)), spk_]
    valid_cell = np.zeros(num_cell).astype('bool')
    cell_shape = np.zeros(num_cell)
    cell_shape[:] = np.nan
    
    for n_cell in range(num_cell):
        if (spk[n_cell].sum()<non_spike_thres) or (spk_[n_cell].sum()<non_spike_thres):
            continue
        spk_shape_b = spk_shape(np.where(spk[n_cell])[0], dff[n_cell])
        spk_shape_a = spk_shape(np.where(spk_[n_cell])[0], dff_[n_cell])        
        var_ = min(spk_shape_b.std(), spk_shape_a.std())
        err_ = np.sqrt(((spk_shape_a-spk_shape_b)**2).mean())
        cell_shape[n_cell] = err_/var_
        if cell_shape[n_cell] < cell_shape_thres:
            valid_cell[n_cell] = True
        if plot_:
            plt.plot(spk_shape_b)
            plt.plot(spk_shape_a)
            var_ = min(spk_shape_b.std(), spk_shape_a.std())
            err_ = np.sqrt(((spk_shape_a-spk_shape_b)**2).mean())
            plt.title(err_/var_)
            plt.show()
    
    np.savez(f'swim_voltr/{folder}_{fish}_swim_voltr_valid_cell', \
            cell_shape=cell_shape, \
            valid_cell=valid_cell)
