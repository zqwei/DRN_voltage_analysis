import numpy as np
import pandas as pd
import os, sys
# from trefide.temporal import TrendFilter
from matplotlib import pyplot as plt
from voltr_spike import *

dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/ProcessedData/'
ablt_len = 6
ablt_sovo_len = 26
# ablt_str = 'after'
# ablt_sovo_str = 'after-swimonly_visualonly'
ablt_sovo_str_alt = '-swimonly_visualonly'

def search_paired_data(row, flist, ablt_len=1, ablt_sovo_str_alt=''):
    if 'before' not in row['fish']:
        return False
    fish = row['fish'][:-ablt_len]
    for _, row_ in flist.iterrows():
        if row_['folder'] != row['folder']:
            continue
        if row_['fish'] == fish+'after'+ablt_sovo_str_alt:
            return True
    return False


def align_components(row, check_shift=False, ablt_len=1, ablt_sovo_str_alt=''):
    from fish_proc.imageRegistration.imTrans import ImAffine
    trans = ImAffine()
    trans.level_iters = [1000, 1000, 100]
    trans.ss_sigma_factor = 1.0
    trans.verbosity = 0
    folder = row['folder']
    fish = row['fish'][:-ablt_len]
    save_fix = dat_folder + f'{folder}/{fish}before{ablt_sovo_str_alt}/Data'
    save_mov = dat_folder + f'{folder}/{fish}after{ablt_sovo_str_alt}/Data'
    # print(save_fix)
    # print(save_mov)
    fix_ = np.load(save_fix + '/motion_fix_.npy').astype('float32')
    move_ = np.load(save_mov + '/motion_fix_.npy').astype('float32')
    # here their shapes are assumed to be identical
    trans_affine = trans.estimate_translation2d(fix_, move_)
    trans_mat = trans_affine.affine
    x, y = trans_mat[0, 2], trans_mat[1, 2]
    x = int(round(x))
    y = int(round(y))
    if check_shift:
        if x>0:
            fix_ = fix_[:-x, :]
            move_ = move_[x:, :]
        else:
            fix_ = fix_[-x:, :]
            move_ = move_[:x, :]
        if y>0:
            fix_ = fix_[:, :-y]
            move_ = move_[:, y:]
        else:
            fix_ = fix_[:, -y:]
            move_ = move_[:, :y]
        plt.imshow(fix_, cmap=plt.cm.Greens)
        plt.imshow(move_, cmap=plt.cm.Reds, alpha=0.5)
        plt.show()
    return x, y


# common code for alignment of components
def shift_xy(fix_, move_, x, y):
    if fix_.ndim ==2:
        fix_ = fix_[:, :, np.newaxis]
    if move_.ndim == 2:
        move_ = move_[:, :, np.newaxis]
    # shift x
    if x>0:
        fix_ = fix_[:-x, :, :]
        move_ = move_[x:, :, :]
    elif x<0:
        fix_ = fix_[-x:, :, :]
        move_ = move_[:x, :, :]
    # shift y
    if y>0:
        fix_ = fix_[:, :-y, :]
        move_ = move_[:, y:, :]
    elif y<0:
        fix_ = fix_[:, -y:, :]
        move_ = move_[:, :y, :]
    return fix_, move_


def shift_xy_before(fix_, x, y):
    if fix_.ndim ==2:
        fix_ = fix_[:, :, np.newaxis]
    # shift x
    if x>0:
        fix_ = fix_[:-x, :, :]
    elif x<0:
        fix_ = fix_[-x:, :, :]
    # shift y
    if y>0:
        fix_ = fix_[:, :-y, :]
    elif y<0:
        fix_ = fix_[:, -y:, :]
    return fix_


def shift_xy_after(move_, x, y):
    if move_.ndim == 2:
        move_ = move_[:, :, np.newaxis]
    # shift x
    if x>0:
        move_ = move_[x:, :, :]
    elif x<0:
        move_ = move_[:x, :, :]
    # shift y
    if y>0:
        move_ = move_[:, y:, :]
    elif y<0:
        move_ = move_[:, :y, :]
    return move_


def get_A_stack(save_folder, is_mask=False, check_stack=False, fext=''):
    import pickle
    
    Y_trend_ave = np.load(f'{save_folder}/Y_trend_ave.npy')
    if is_mask:
        _ = np.load(f'{save_folder}/mask.npz')
        mask = _['mask']
        mask_save = _['mask_save']
        Y_trend_ave = Y_trend_ave[mask_save[0].min():mask_save[0].max(), mask_save[1].min():mask_save[1].max()]
    with open(f'{save_folder}/period_Y_demix{fext}_rlt.pkl', 'rb') as f:
        rlt_ = pickle.load(f)
    d1, d2 = Y_trend_ave.shape

    mask_ = np.empty((d2, d1))
    mask_[:] = False
    if is_mask:
        pixel = 1
    else:
        pixel = 4
    mask_[:pixel, :]=True
    mask_[-pixel:,:]=True
    mask_[:, :pixel]=True
    mask_[:,-pixel:]=True
    mask_ = mask_.astype('bool')
    A = rlt_['fin_rlt']['a'].copy()
    # remove bounds at pixel size
    A[mask_.reshape(-1),:]=0

    # remove small size components
    A_ = A[:, (A>0).sum(axis=0)>40] # min pixel = 40

    # remove stripes
    # cherry-pick Components
    remove_comp = np.empty(A_.shape[-1]).astype('bool')
    remove_comp[:] = False
    for n_, nA_ in enumerate(A_.T):
        x_, y_ = np.where(nA_.reshape(d2, d1).T>0)
        len_x = x_.max()-x_.min()
        len_y = y_.max()-y_.min()
        if len_x==0 or len_y==0 or len_x/len_y>10 or len_y/len_x>10 or len_x<=3 or len_y<=3:
            remove_comp[n_] = True
    A_ = A_[:, ~remove_comp]
    d1, d2 = Y_trend_ave.shape
    A_tmp = []
    for n, nA in enumerate(A_.T):
        nA = nA.reshape(d2, d1).T
        A_tmp.append(nA)
    A_tmp = np.array(A_tmp).transpose([1, 2, 0])
    assert np.array_equal(A_tmp.reshape((d1*d2, -1), order='F'), A_) # make sure the array is correctly reshaped
    
    if not is_mask:
        A_stack = A_tmp
    else:
        # add mask back to A_stack
        A_stack = np.zeros((mask.shape[0], mask.shape[1], A_tmp.shape[-1]))
        for n in range(A_tmp.shape[-1]):
            A_stack[mask_save[0].min():mask_save[0].max(), mask_save[1].min():mask_save[1].max(), n] = A_tmp[:, :, n]
    if check_stack:
        Y_trend_ave = np.load(f'{save_folder}/Y_trend_ave.npy')
        _ = A_stack.reshape((Y_trend_ave.shape[0]*Y_trend_ave.shape[1], -1), order='F')
        plot_components(_, Y_trend_ave)
    return A_stack


def get_C_stacks(save_folder, pix_x, pix_y, A_, is_before=True, is_mask=True, fext=''):
    from pathlib import Path
    from skimage.external.tifffile import imread
    import pickle
    from fish_proc.utils.demix import recompute_C_matrix, pos_sig_correction
    
    Y_trend_ave = np.load(f'{save_folder}/Y_trend_ave.npy')
    _ = np.load(f'{save_folder}/Y_2dnorm.npz')
    Y_d_std= _['Y_d_std']
    mov = -imread(f'{save_folder}/Y_svd.tif').astype('float32')*Y_d_std
    with open(f'{save_folder}/period_Y_demix{fext}_rlt.pkl', 'rb') as f:
        rlt_ = pickle.load(f)
    
    b = rlt_['fin_rlt']['b']
    fb = rlt_['fin_rlt']['fb']
    ff = rlt_['fin_rlt']['ff']
    dims = mov.shape
    if fb is not None:
        b_tmp = np.matmul(fb, ff.T)+b
    else:
        b_tmp = b
    
    if not is_mask:
        b_ = b_tmp.reshape((dims[0], dims[1], -1), order='F')
    else:
        b_ = np.zeros((dims[0], dims[1], b_tmp.shape[-1]), order='F')
        _ = np.load(f'{save_folder}/mask.npz')
        mask = _['mask']
        mask_save = _['mask_save']
        d1 = mask_save[0].max()-mask_save[0].min()
        d2 = mask_save[1].max()-mask_save[1].min()
        assert (d1*d2) == b_tmp.shape[0]
        for n in range(b_.shape[-1]):
            b_[mask_save[0].min():mask_save[0].max(), mask_save[1].min():mask_save[1].max(), n] = b_tmp[:, n].reshape((d1, d2), order='F')

    if is_before:
        mov = shift_xy_before(mov, pix_x, pix_y)
        b_ = shift_xy_before(b_, pix_x, pix_y)
        Y_trend_ave = shift_xy_before(Y_trend_ave, pix_x, pix_y)
    else:
        mov = shift_xy_after(mov, pix_x, pix_y)
        b_ = shift_xy_after(b_, pix_x, pix_y)
        Y_trend_ave = shift_xy_after(Y_trend_ave, pix_x, pix_y)
    
    mov = pos_sig_correction(mov, -1)
    mov = mov - b_
    C_ = recompute_C_matrix(mov, A_)
    base_ = recompute_C_matrix(Y_trend_ave, A_)
    np.savez_compressed(f'{save_folder}/Voltr_raw{fext}', A_=A_, C_=C_, base_=base_)
    Path(save_folder+f'/finished_voltr{fext}.tmp').touch()
    return None


def voltron_ablt(row, pix_x, pix_y, fext='', is_mask=False, ablt_len=1, ablt_sovo_str_alt=''):
    from pathlib import Path
    folder = row['folder']
    fish = row['fish'][:-ablt_len]
    save_folder_before = dat_folder + f'{folder}/{fish}before{ablt_sovo_str_alt}/Data'
    save_image_folder_before = dat_folder + f'{folder}/{fish}before{ablt_sovo_str_alt}/Results'
    save_folder_after = dat_folder + f'{folder}/{fish}after{ablt_sovo_str_alt}/Data'
    save_image_folder_after = dat_folder + f'{folder}/{fish}after{ablt_sovo_str_alt}/Results'

    if not os.path.exists(save_image_folder_before):
        os.makedirs(save_image_folder_before)
    if not os.path.exists(save_image_folder_after):
        os.makedirs(save_image_folder_after)
    print('=====================================')
    print(save_folder_before)
    print(save_folder_after)

    if os.path.isfile(save_folder_after+f'/finished_voltr{fext}.tmp'):
        return None

    if not os.path.isfile(f'{save_folder_before}/period_Y_demix{fext}_rlt.pkl'):
        print('Components file does not exist.')
        return None
    if not os.path.isfile(f'{save_folder_after}/period_Y_demix{fext}_rlt.pkl'):
        print('Components file does not exist.')
        return None

    if os.path.isfile(save_folder_before+f'/proc_voltr{fext}.tmp'):
        print('File is already in processing.')
        return None

    Path(save_folder_before+f'/proc_voltr{fext}.tmp').touch()
    Path(save_folder_after+f'/proc_voltr{fext}.tmp').touch()

    print('Combining the components in before and after ablation')

    A_before = get_A_stack(save_folder_before, is_mask=True, check_stack=False, fext=fext)
    A_after = get_A_stack(save_folder_after, is_mask=True, check_stack=False, fext=fext)
    A_before, A_after = shift_xy(A_before, A_after, pix_x, pix_y)
    d1, d2, _ = A_before.shape
    A_ = np.concatenate((A_before, A_after), axis=-1)
    A_ = A_.reshape((d1*d2, -1), order='F')
    A_ = A_[:, A_.sum(axis=0)>0] # remove zeros-sum components
    
    Y_trend_before = np.load(f'{save_folder_before}/Y_trend_ave.npy')
    Y_trend_after = np.load(f'{save_folder_after}/Y_trend_ave.npy')
    Y_trend_before, Y_trend_after = shift_xy(Y_trend_before, Y_trend_after, pix_x, pix_y)
    
    plot_components(A_, Y_trend_before.squeeze(-1), fext=fext, save_image_folder=save_image_folder_before)
    plot_components(A_, Y_trend_after.squeeze(-1), fext=fext, save_image_folder=save_image_folder_after)

    print('Start computing voltron data')
    get_C_stacks(save_folder_before, pix_x, pix_y, A_, is_before=True, is_mask=True, fext=fext)
    get_C_stacks(save_folder_after, pix_x, pix_y, A_, is_before=False, is_mask=True, fext=fext)

    return None