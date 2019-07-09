#!/groups/ahrens/home/weiz/anaconda3/envs/myenv/bin/python

import numpy as np
import pandas as pd
import os
from pathlib import Path
from nmf_calcium import *
from utils_swim import *

vol_file = '../Voltron_data/SnFR_Log_DRN_Exp.csv'
dat_xls_file = pd.read_csv(vol_file)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/SnFRData/'
dir_folder = Path(dat_folder)


for index, row in dat_xls_file.iterrows():
    if row['type'] != 'tph2':
        continue
    folder = row['folder']
    fish = row['fish']
    rootDir = row['rootDir']
    dff_dir = dat_folder+f'{folder}/{fish}/Data/'
    swm_dir = dat_folder+f'{folder}/{fish}/swim/'
    img_dir = rootDir+f'{folder}/{fish}/Registered'
    
    if not os.path.exists(swm_dir+'frame_swim_tcourse_series.npy'):
        swim(folder, fish, rootDir, dat_folder)
        trial_swim_power(folder, fish, dir_folder)
        frame_swim_power(folder, fish, dir_folder)
        frame_swim_power_series(folder, fish, dir_folder)
    
    
    ave = np.load(img_dir+'/stack_sub.npy')[()]
    np.save(dff_dir+'ave_img.npy', ave.mean(axis=0))
    
    if not os.path.exists(dff_dir):
        os.makedirs(dff_dir)
    if os.path.exists(dff_dir+'components.npz'):
        continue
    if os.path.exists(dff_dir+'processing.tmp'):
        continue
    
    open(dff_dir+'processing.tmp', 'a').close()
    print(dff_dir)
    dFF = np.load(img_dir+'/dFF_sub.npy')[()]
    dFF = dFF.transpose((1, 2, 0))
    dFF = denoise_sig(dFF)
    demix_components(dFF, dff_dir)
    
    os.remove(dff_dir+'processing.tmp')
    
    