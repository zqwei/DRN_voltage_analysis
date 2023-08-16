import numpy as np
import pandas as pd
from skimage.io import imread
from utils import *
from skimage.exposure import equalize_adapthist as clahe
from skimage.morphology import square, dilation
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.morphology import dilation
import os

vol_file = '../SnFR_data/SnFR_Log_DRN_Exp.csv'
dat_xls_file = pd.read_csv(vol_file)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/SnFRData/'
dat_folder_ = '/nearline/ahrens/Ziqiang_tmp/SnFRData/'

## save component df/f


for index, row in dat_xls_file.iterrows():
    folder = row['folder']
    fish = row['fish']
    if os.path.exists(f'snfr_dff_simple/{folder}_{fish}_snfr_dff_dat.npz'):
        continue
    dff_dir = dat_folder+f'{folder}/{fish}/Data/'
    dff_dir_= dat_folder_+f'{folder}/{fish}/Data/'
    A = np.load(dff_dir+'components.npz', allow_pickle=True)['A_']
    Y_ = imread(dff_dir_+'imgDMotion.tif')
    t, x, y = Y_.shape
    
    ## save average image df/f
    ave_img = Y_.mean(axis=0)
    valid_pix = ave_img>20
    F = Y_[:, valid_pix].mean(axis=-1)
    F_b = baseline(F, window=1000, percentile=20)
    dFF_ave = F/F_b-1
    dFF_ave[:100] = 0
    F_mean=Y_.mean(axis=0)
    
    block_img = ave_img.copy()
    block_img[ave_img<20]=0
    img_adapteq = clahe(block_img/block_img.max(), clip_limit=0.03)
    segments_fz = felzenszwalb(img_adapteq, scale=100, sigma=0.5, min_size=3000)
    seg = segments_fz.copy()+1
    seg[~valid_pix]=0
    dFF_ = np.zeros((t, seg.max()))
    seg_list=[]
    for n in range(seg.max()):
        seg_pix = seg==(n+1)
        if seg_pix.sum()<3000:
            continue
        F = Y_[:, seg_pix].mean(axis=-1)
        F_b = baseline(F, window=1000, percentile=20)
        _ = F/F_b-1
        _[:100] = 0
        dFF_[:, n] = _
        seg_list.append(seg_pix)
    dFF_ = dFF_[:, np.abs(dFF_).sum(axis=0)>0]
    
    np.savez(f'snfr_dff_simple/{folder}_{fish}_snfr_dff_dat', \
        dFF_ave=dFF_ave, A=seg_list, dFF=dFF_, Y_mean=F_mean)
    