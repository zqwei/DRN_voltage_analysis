import numpy as np
import pandas as pd
from skimage.io import imread
from utils import *

vol_file = '../SnFR_data/SnFR_Log_DRN_Exp.csv'
dat_xls_file = pd.read_csv(vol_file)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/SnFRData/'

## save component df/f


for index, row in dat_xls_file.iterrows():
    folder = row['folder']
    fish = row['fish']
    dff_dir = dat_folder+f'{folder}/{fish}/Data/'
    A = np.load(dff_dir+'components.npz', allow_pickle=True)['A_']
    Y_ = imread(dff_dir+'imgDMotion.tif')
    t, x, y = Y_.shape
    
    ## save average image df/f
    valid_pix = Y_.mean(axis=0)>100
    F = Y_[:, valid_pix].mean(axis=-1)
    F_b = baseline(F, window=1000, percentile=20)
    dFF_ave = F/F_b-1
    dFF_ave[:100] = 0
    F_mean=Y_.mean(axis=0)
    
    A_new = A.copy()
    for n, a_ in enumerate(A.T):
        _ = a_
        _[a_<a_.max()*.2] = 0
        A_new[:, n]=_
    F = np.matmul(Y_.reshape(t, x*y, order='F'), A_new)
    F_b = baseline(F, window=1000, percentile=20, axis=0)
    dFF = F/F_b-1
    dFF[:100, :] = 0
    
    np.savez(f'snfr_dff/{folder}_{fish}_snfr_dff_dat', \
        dFF_ave=dFF_ave, A=A_new, F=F, dFF=dFF, Y_mean=F_mean)
    