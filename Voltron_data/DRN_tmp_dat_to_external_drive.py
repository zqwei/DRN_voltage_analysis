import numpy as np
import pandas as pd
import os, shutil

dat_folder = '/Volumes/ahrens/Ziqiang/Takashi_DRN_project/ProcessedData/'
target_folder = '/Volumes/Ahrens_lab_data_vol_01/Takashi_DRN_project/ProcessedData/'
dat_xls_file = pd.read_csv('Voltron_Log_DRN_Exp_moved_to_tapes.csv', index_col=0)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')

for index, row in dat_xls_file.iterrows():
    folder = row['folder']
    fish = row['fish']
    if not row['Plan']:
        continue
    if row['moved']:
        continue
    save_folder = dat_folder + f'{folder}/{fish}/Data/'
    move_folder = target_folder + f'{folder}/{fish}/Data/'
    if not os.path.exists(move_folder):
        os.makedirs(move_folder)
    if not os.path.exists(save_folder + 'imgDNoMotion.tif'):
        print(f'No pixel denoise data at {save_folder}')
    else:
        shutil.move(save_folder + 'imgDNoMotion.tif', move_folder + 'imgDNoMotion.tif')
        print(f'Pixel denoise data moved from {save_folder}')
    if not os.path.isfile(save_folder+'imgDMotion.tif'):
        print(f'No motion correction data at {save_folder}')
    else:
        shutil.move(save_folder + 'imgDMotion.tif', move_folder + 'imgDMotion.tif')
        print(f'Motion correction data moved from {save_folder}')
    if not os.path.isfile(save_folder+'Y_d.tif'):
        print(f'No detrend data at {save_folder}')
    else:
        shutil.move(save_folder + 'Y_d.tif', move_folder + 'Y_d.tif')
        print(f'Detrend data moved from {save_folder}')
    if not os.path.isfile(save_folder+'Y_svd.tif'):
        print(f'No denoised data at {save_folder}')
    else:
        shutil.move(save_folder + 'Y_svd.tif', move_folder + 'Y_svd.tif')
        print(f'Denoised data moved from {save_folder}')
