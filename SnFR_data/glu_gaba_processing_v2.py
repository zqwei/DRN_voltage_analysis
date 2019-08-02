#!/groups/ahrens/home/weiz/miniconda3/envs/myenv/bin/python

import numpy as np
import pandas as pd
import os
from pathlib import Path
from preprocess_calcium import *
from utils_swim import *

vol_file = '../SnFR_data/SnFR_Log_DRN_Exp_v2.csv'
dat_xls_file = pd.read_csv(vol_file)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/SnFRData/'
dir_folder = Path(dat_folder)


for index, row in dat_xls_file.iterrows():
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
    pixel_denoise(row)
    registration(row)
    video_detrend(row)
#     local_pca_demix(row)
