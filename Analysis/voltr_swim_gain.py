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
vol_file = 'depreciated/analysis_sections_gain.csv'
dat_xls_file = pd.read_csv(vol_file, index_col=0)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')

t_pre = 100 # time window pre-swim
t_post = 350 # time window post-swim
t_len = t_pre+t_post
t_sig = 300 # time used for significance test after swim
non_spike_thres = 100
k_spk = boxcarKernel(sigma=61)
k_sub = 10

for _, row in dat_xls_file.iterrows():
    folder = row['folder']
    fish = row['fish']
    dat_dir = dir_folder+f'{folder}/{fish}/Data/'
    dff = np.load(dat_dir+'Voltr_spikes.npz')['voltrs']
    spk = np.load(dat_dir+'Voltr_spikes.npz')['spk']
    dff = dff - np.nanmedian(dff, axis=1, keepdims=True)
    num_cell = spk.shape[0]
    spk = np.r_['-1', np.zeros((num_cell, 600)), spk]

    _ = np.load(f'swim_power/{folder}_{fish}_swim_dat.npz')
    swim_starts = _['swim_starts']
    swim_ends = _['swim_ends']
    r_swim = _['r_swim']
    l_swim = _['l_swim']
    visu = _['visu']
    task_period = _['task_period']
    swim_task_index = _['swim_task_index']
    p_swim = np.sqrt(r_swim**2+l_swim**2)

    n_task = task_period.max().astype('int')
    swim_task_index_ = (swim_task_index-1)%n_task+1
    swim_bout_index_task = np.zeros(len(swim_task_index_)).astype('int')
    swim_bout_index_task[0] = 1
    for n_bout in range(len(swim_task_index_)-1):
        if swim_task_index_[n_bout] == swim_task_index_[n_bout-1]:
            swim_bout_index_task[n_bout] = swim_bout_index_task[n_bout-1]+1
        else:
            swim_bout_index_task[n_bout] = 1
    
    num_cell = dff.shape[0]
    
    trial_valid_ = np.ones(len(swim_starts)).astype('bool')
    for n, n_swim in enumerate(swim_starts[:-1]):        
        # examine the swim with short inter-swim-interval
        if swim_starts[n+1] - n_swim < t_sig:    
            trial_valid_[n] = False
    
    sub_swim = []
    spk_swim = []
    raw_spk_swim=[]
    sub_sig_swim = []
    # remove low spike cells
    for n_cell in range(num_cell):
        if spk[n_cell].sum()<non_spike_thres:
            continue
        n_spk = smooth(spk[n_cell], k_spk)
        n_dff = medfilt(dff[n_cell], kernel_size=k_sub*2+1)
        
        sub_list = np.empty((r_swim.shape[0], t_len))
        sub_list[:] = np.nan
        sub_list_ = np.zeros((r_swim.shape[0], t_len))
        spk_list = np.empty((r_swim.shape[0], t_len))
        spk_list[:] = np.nan
        raw_spk_list = np.empty((r_swim.shape[0], t_len))
        raw_spk_list[:] = np.nan
        sub_sig = np.ones(t_sig)
        
        for n, n_swim in enumerate(swim_starts):
            if (n_swim>t_pre) and (n_swim+t_post<len(n_dff)):
                # get sub and spk for each swim
                sub_list[n, :] = n_dff[n_swim-t_pre:n_swim+t_post] 
                sub_list_[n, :] = sub_list[n, :] - sub_list[n, (t_pre-30):t_pre].mean()
                spk_list[n, :] = n_spk[n_swim-t_pre:n_swim+t_post]
                raw_spk_list[n, :] = spk[n_cell][n_swim-t_pre:n_swim+t_post]
            else:
                trial_valid_[n] = False
            
        ave_low = sub_list_[(task_period==1) & trial_valid_, :]
        ave_high = sub_list_[(task_period==2) & trial_valid_, :]

        if (ave_low.shape[0]>10) and (ave_high.shape[0]>10):
            for _ in range(t_sig):
                __, sub_sig[_] = ranksums(ave_low[:, _+t_pre], ave_high[:, _+t_pre])
        
        sub_swim.append(sub_list)
        spk_swim.append(spk_list)
        sub_sig_swim.append(sub_sig)
        raw_spk_swim.append(raw_spk_list)
    
    np.savez(f'swim_voltr/{folder}_{fish}_swim_voltr_dat', \
            sub_swim=np.array(sub_swim), \
            spk_swim=np.array(spk_swim), \
            raw_spk_swim=np.array(raw_spk_swim), \
            sub_sig_swim=np.array(sub_sig_swim), \
            trial_valid=trial_valid_)
