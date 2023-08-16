import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
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

vol_file = '../SnFR_data/SnFR_Log_DRN_Exp_new.csv'
dat_xls_file = pd.read_csv(vol_file)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
dat_folder = '/scratch/weiz/Takashi_DRN_project/SnFRData/'

for index, row in dat_xls_file.iterrows():
    folder = row['folder']
    fish = row['fish']
    if os.path.exists(f'snfr_dff_simple_center/{folder}_{fish}_snfr_dff_dat.npz'):
        continue
    dff_dir = dat_folder+f'{folder}/{fish}/Data/'
    img_ = imread(dff_dir+'imgDMotion.tif')
    
    ave_img = img_.mean(axis=0)
    thres = 20
    mask = ave_img>thres
    mask_ = ave_img>np.percentile(ave_img, 90)

    segments_fz = felzenszwalb(mask_, scale=150, sigma=0.7, min_size=6000)

    X_, Y_ = np.meshgrid(np.arange(ave_img.shape[1]),np.arange(ave_img.shape[0]))

    for n in range(1, segments_fz.max()+1):
        cx = X_[segments_fz==n].mean()
        cy = Y_[segments_fz==n].mean()
        if (ave_img.shape[0]/3)<cy<(ave_img.shape[0]/3*2):
            mask[segments_fz==n]=False
    
#     plt.figure(figsize=(4,8))
#     plt.imshow(ave_img)
#     plt.imshow(mask.astype('int'), cmap=plt.cm.gray, alpha=0.5)
#     plt.savefig(f"snfr_dff_simple_center/ave_img_{folder}_{fish}.png")
#     plt.close('all')
    
    F = img_[:, mask].mean(axis=-1)
    F_b = baseline(F, window=1000, percentile=20)
    dFF_ave = F/F_b-1
    dFF_ave[:100] = 0
    
    np.savez(f'snfr_dff_simple_center/{folder}_{fish}_snfr_dff_dat', \
        dFF_ave=dFF_ave, A=mask, Y_mean=ave_img)

    
    