#!/groups/ahrens/home/weiz/miniconda3/envs/myenv/bin/python

import numpy as np
import pandas as pd
import os
from pathlib import Path

vol_file = '../SnFR_data/SnFR_Log_DRN_Exp.csv'
dat_xls_file = pd.read_csv(vol_file, index_col=0)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/SnFRData/'

for index, row in dat_xls_file.iterrows():
    if row['type'] != 'tph2':
        continue
    folder = row['folder']
    fish = row['fish']
    save_folder = dat_folder + f'{folder}/{fish}/'
    if os.path.isfile(save_folder + 'Data/motion_fix_.npy'):
        dat_xls_file.at[index, 'pixeldenoise'] = True
    if os.path.isfile(save_folder+'Data/finished_registr.tmp'):
        dat_xls_file.at[index, 'registration'] = True
    if os.path.isfile(save_folder+'Data/finished_detrend.tmp'):
        dat_xls_file.at[index, 'detrend'] = True
    if os.path.isfile(save_folder+'Data/Y_local.npz'):
        dat_xls_file.at[index, 'denoise'] = True
    if os.path.isfile(save_folder+'Data/finished_local_denoise_demix.tmp'):
        dat_xls_file.at[index, 'demix'] = True
print(dat_xls_file.sum(numeric_only=True))
dat_xls_file.to_csv(vol_file)
