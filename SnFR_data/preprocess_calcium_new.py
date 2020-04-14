import numpy as np
import os, sys
from fish_proc.utils.memory import get_process_memory, clear_variables
from pathlib import Path
from nmf_calcium import *
# dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/SnFRData/'
dat_folder = '/scratch/weiz/Takashi_DRN_project/SnFRData/'
cameraNoiseMat = '/nrs/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'


def remove_tmp_files(row):
    from glob import glob
    folder = row['folder']
    fish = row['fish']
    image_folder = row['rootDir'] + f'{folder}/{fish}/'
    save_folder = dat_folder + f'{folder}/{fish}/Data'
    flist = glob(save_folder+'/*.tmp')
    for f in flist:
        try:
            os.remove(f)
        except:
            print("Error while deleting file : ", f)

            
def pixel_denoise(row):
    from fish_proc.pipeline.preprocess import pixel_denoise, pixel_denoise_img_seq
    folder = row['folder']
    fish = row['fish']
    image_folder = row['rootDir'] + f'{folder}/{fish}/'
    save_folder = dat_folder + f'{folder}/{fish}/Data'
    if os.path.exists(save_folder+'/imgDNoMotion.tif'):
        return None
    print(save_folder)
    if os.path.exists(image_folder+'Registered/raw.tif'):
        imgD_ = pixel_denoise(image_folder, 'Registered/raw.tif', save_folder, cameraNoiseMat, plot_en=False)
    else:
        imgD_ = pixel_denoise_img_seq(image_folder, save_folder, cameraNoiseMat, plot_en=False)
    get_process_memory();
    imgD_ = None
    fix_ = None
    clear_variables((imgD_, fix_))
    return None


def registration(row, is_largefile=False):
    '''
    Generate imgDMotion.tif
    '''
    from skimage.io import imread, imsave
    from tqdm import tqdm
    folder = row['folder']
    fish = row['fish']
    save_folder = dat_folder + f'{folder}/{fish}/Data'
    if os.path.exists(save_folder+'/imgDMotion.tif'):
        return None
    imgD_ = imread(save_folder+'/imgDNoMotion.tif').astype('float32')
    affs = np.load(save_folder + '/imgDMotionVar.npy').astype('float32')
    imgDMotion=imgD_.copy()
    for ntime in tqdm(range(affs.shape[0])):
        move_=affs[ntime]
        mov=imgD_[ntime]
        motion_correction_image(mov, move_)
        imgDMotion[ntime]=motion_correction_image(mov, move_)
    imsave(save_folder+'/imgDMotion.tif', imgDMotion, compress=1)
    return None


def motion_correction_image(mov, move_):
    from scipy.ndimage.interpolation import affine_transform
    affs = np.eye(3)
    affs[0, 2]=move_[1]
    affs[1, 2]=move_[2]
    return affine_transform(mov, affs)
