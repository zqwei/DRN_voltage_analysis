import numpy as np
import os, sys
from fish_proc.utils.memory import get_process_memory, clear_variables
from pathlib import Path
from nmf_calcium import *
dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/SnFRData/'
cameraNoiseMat = '/nrs/ahrens/ahrenslab/Ziqiang/gainMat/gainMat20180208'


def pixel_denoise(row):
    from fish_proc.pipeline.preprocess import pixel_denoise, pixel_denoise_img_seq
    folder = row['folder']
    fish = row['fish']
    image_folder = row['rootDir'] + f'{folder}/{fish}/'
    save_folder = dat_folder + f'{folder}/{fish}/Data'
    if os.path.exists(image_folder):
        print(f'checking file {folder}/{fish}')
        if not os.path.exists(save_folder+'/'):
            os.makedirs(save_folder)
        if os.path.exists(save_folder + 'imgDNoMotion.tif'):
            return None
        if os.path.exists(save_folder+'/finished_pixel_denoise.tmp'):
            return None
        if os.path.exists(save_folder+'/proc_pixel_denoise.tmp'):
            return None
        if not os.path.isfile(save_folder + '/motion_fix_.npy'):
            print(f'process file {folder}/{fish}')
            try:
                Path(save_folder+'/proc_pixel_denoise.tmp').touch()
                if os.path.exists(image_folder+'Registered/raw.tif'):
                    imgD_ = pixel_denoise(image_folder, 'Registered/raw.tif', save_folder, cameraNoiseMat, plot_en=True)
                else:
                    imgD_ = pixel_denoise_img_seq(image_folder, save_folder, cameraNoiseMat, plot_en=True)
                t_ = len(imgD_)//2
                win_ = 150
                fix_ = imgD_[t_-win_:t_+win_].mean(axis=0)
                np.save(save_folder + '/motion_fix_', fix_)
                get_process_memory();
                imgD_ = None
                fix_ = None
                clear_variables((imgD_, fix_))
                Path(save_folder+'/finished_pixel_denoise.tmp').touch()
            except MemoryError as err:
                print(f'Memory Error on file {folder}/{fish}: {err}')
                os.remove(save_folder+'/proc_pixel_denoise.tmp')
    return None


def registration(row, is_largefile=False):
    '''
    Generate imgDMotion.tif
    '''
    from fish_proc.pipeline.preprocess import motion_correction
    from skimage.io import imread, imsave
    folder = row['folder']
    fish = row['fish']
    save_folder = dat_folder + f'{folder}/{fish}/Data'
    print(f'checking file {folder}/{fish}')
    if os.path.isfile(save_folder+'/finished_registr.tmp'):
        return None
    if os.path.isfile(save_folder+'/finished_detrend.tmp'):
        Path(save_folder+'/finished_registr.tmp').touch()
        return None
    if not os.path.isfile(save_folder+'/imgDMotion.tif') and os.path.isfile(save_folder + '/motion_fix_.npy'):
        if not os.path.isfile(save_folder+'/proc_registr.tmp'):
            try:
                Path(save_folder+'/proc_registr.tmp').touch()
                print(f'process file {folder}/{fish}')
                imgD_ = imread(save_folder+'/imgDNoMotion.tif').astype('float32')
                fix_ = np.load(save_folder + '/motion_fix_.npy').astype('float32')
                if is_largefile:
                    len_D_ = len(imgD_)//2
                    motion_correction(imgD_[:len_D_], fix_, save_folder, ext='0')
                    get_process_memory();
                    motion_correction(imgD_[len_D_:], fix_, save_folder, ext='1')
                    get_process_memory();
                    imgD_ = None
                    fix_ = None
                    clear_variables((imgD_, fix_))
                    s_ = [np.load(save_folder+'/imgDMotion%d.npy'%(__)) for __ in range(2)]
                    s_ = np.concatenate(s_, axis=0).astype('float32')
                    imsave(save_folder+'/imgDMotion.tif', s_, compress=1)
                    s_ = None
                    clear_variables(s_)
                    os.remove(save_folder+'/imgDMotion0.npy')
                    os.remove(save_folder+'/imgDMotion1.npy')
                else:
                    motion_correction(imgD_, fix_, save_folder)
                    get_process_memory();
                    imgD_ = None
                    fix_ = None
                    clear_variables((imgD_, fix_))
                    s_ = np.load(save_folder+'/imgDMotion.npy').astype('float32')
                    imsave(save_folder+'/imgDMotion.tif', s_, compress=1)
                    s_ = None
                    clear_variables(s_)
                    os.remove(save_folder+'/imgDMotion.npy')
                Path(save_folder+'/finished_registr.tmp').touch()
            except Exception as err:
                print(f'Registration failed on file {folder}/{fish}: {err}')
                os.remove(save_folder+'/proc_registr.tmp')
    return None


def video_detrend(row):
    from fish_proc.pipeline.denoise import detrend
    from multiprocessing import cpu_count
    from skimage.io import imsave, imread

    folder = row['folder']
    fish = row['fish']
    save_folder = dat_folder + f'{folder}/{fish}/Data'
    print(f'checking file {folder}/{fish}')
    if os.path.isfile(save_folder+'/finished_detrend.tmp'):
        return None

    if not os.path.isfile(save_folder+'/Y_d.tif') and not os.path.isfile(save_folder+'/proc_detrend.tmp'):
        if os.path.isfile(save_folder+'/finished_registr.tmp'):
            try:
                Path(save_folder+'/proc_detrend.tmp').touch()
                Y = imread(save_folder+'/imgDMotion.tif').astype('float32')
                Y = Y.transpose([1,2,0])
                n_split = min(Y.shape[0]//cpu_count(), 8)
                if n_split <= 1:
                    n_split = 2
                Y_len = Y.shape[0]//2
                detrend(Y[:Y_len, :, :], save_folder, n_split=n_split//2, ext='0')
                detrend(Y[Y_len:, :, :], save_folder, n_split=n_split//2, ext='1')
                Y = None
                clear_variables(Y)
                get_process_memory();
                Y = []
                Y.append(np.load(save_folder+'/Y_d0.npy').astype('float32'))
                Y.append(np.load(save_folder+'/Y_d1.npy').astype('float32'))
                Y = np.concatenate(Y, axis=0).astype('float32')
                imsave(save_folder+'/Y_d.tif', Y, compress=1)
                Y = None
                clear_variables(Y)
                get_process_memory();
                os.remove(save_folder+'/Y_d0.npy')
                os.remove(save_folder+'/Y_d1.npy')
                Path(save_folder+'/finished_detrend.tmp').touch()
            except Exception as err:
                print(f'Detrend failed on file {folder}/{fish}: {err}')
                os.remove(save_folder+'/proc_detrend.tmp')
    return None


def local_pca_demix(row):
    folder = row['folder']
    fish = row['fish']
    save_folder = dat_folder + f'{folder}/{fish}/Data'
    print(f'checking file {folder}/{fish}')
    if os.path.isfile(save_folder+'/finished_local_denoise.tmp'):
        return None

    if not os.path.isfile(save_folder+'/proc_local_denoise_demix.tmp'):
        if os.path.isfile(save_folder+'/finished_detrend.tmp'):
            try:
                Path(save_folder+'/proc_local_denoise.tmp').touch()
                if os.path.isfile(f'{save_folder}/Y_d.npy'):
                    Y_d = np.load(f'{save_folder}/Y_d.npy').astype('float32')
                elif os.path.isfile(f'{save_folder}/Y_d.tif'):
                    Y_d = imread(f'{save_folder}/Y_d.tif')
                Y_d_ave = Y_d.mean(axis=-1, keepdims=True) # remove mean
                Y_d_std = Y_d.std(axis=-1, keepdims=True) # normalization
                Y_d = (Y_d - Y_d_ave)/Y_d_std
                Y_d = Y_d.astype('float32')
                np.savez_compressed(f'{save_folder}/Y_2dnorm', Y_d_ave=Y_d_ave, Y_d_std=Y_d_std)
                Y_d_ave = None
                Y_d_std = None
                clear_variables((Y_d_ave, Y_d_std))
                dFF = denoise_sig(Y_d)
                demix_components(dFF, save_folder)
                get_process_memory();
                Path(save_folder+'/finished_local_denoise_demix.tmp').touch()
            except Exception as err:
                print(f'Local pca and demix failed on file {folder}/{fish}: {err}')
                os.remove(save_folder+'/proc_local_denoise_demix.tmp')
    return None
