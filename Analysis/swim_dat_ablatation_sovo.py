"""
This file supplies the functions to identify valid swims.

Valid swim is defined as the fish adapts to the gain

Created on 03/21/2019
@author: Ziqiang Wei
@email: weiz@janelia.hhmi.org
"""

import numpy as np
import pandas as pd
import os

vol_file = '../Voltron_data/Voltron_Log_DRN_Exp_update.csv'
dat_xls_file = pd.read_csv(vol_file, index_col=0)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
dir_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/ProcessedData/'
t_pre = 100 # time window pre-swim
t_post = 350 # time window post-swim
t_len = t_pre+t_post
gain_stat_len = 100 # time length to examine the gain adaption after swim
before_ = 'before-swimonly_visualonly'
after_ = 'after-swimonly_visualonly'
before_len = len(before_)
after_len = len(after_)

def search_paired_data_before(row, flist):
    if 'before ablation' not in row['task']:
        return False
    if 'failed' in row['task']:
        return False
    fish = row['fish'][:-before_len]
    for _, row_ in flist.iterrows():
        if row_['folder'] != row['folder']:
            continue
        if row_['fish'] == (fish+after_):
            return True
    return False


def search_paired_data_after(row, flist):
    if 'after ablation' not in row['task']:
        return False
    if 'failed' in row['task']:
        return False
    fish = row['fish'][:-after_len]
    for _, row_ in flist.iterrows():
        if row_['folder'] != row['folder']:
            continue
        if row_['fish'] == (fish+before_):
            return True
    return False


def valid_swim(row):
    from scipy.stats import ranksums
    folder = row['folder']
    fish = row['fish']
    task_type = row['task'] # task type names
    if not 'ablation' in task_type:
        return False
    if not 'Swimonly_Visualonly' in task_type:
        return False
    
    if not (search_paired_data_before(row, dat_xls_file) or search_paired_data_after(row, dat_xls_file)):
        return False

    swim_dir = dir_folder + f'{folder}/{fish}/swim/'
    print(swim_dir)
    frame_stimParams = np.load(swim_dir+'frame_stimParams.npy')
    frame_swim_tcourse = np.load(swim_dir+'frame_swim_tcourse_series.npy')
    rawdata = np.load(swim_dir+"rawdata.npy", allow_pickle=True)[()]
    swimdata = np.load(swim_dir+"swimdata.npy", allow_pickle=True)[()]
    reclen=len(swimdata['fltCh1'])
    frame_tcourse=np.zeros((reclen,))
    frame=np.where(np.diff((rawdata['ch3']>3).astype('int'))==1)[0]+1
    task_tcourse = np.zeros(len(frame))
    for t in range(len(frame)-1):
        frame_tcourse[frame[t]:frame[t+1]]=t
        task_tcourse[t] = rawdata['stimParam4'][frame[t+1]]
    swim_starts = frame_tcourse[np.where(swimdata['swimStartT']>0)[0]].astype('int')
    swim_ends = frame_tcourse[np.where(swimdata['swimEndT']>0)[0]].astype('int')
    # collect trial within t-pre, and t-post valid range
    swim_ends   = swim_ends[((swim_starts>t_pre) & (swim_starts<(frame_swim_tcourse.shape[1]-t_post)))]
    swim_starts = swim_starts[((swim_starts>t_pre) & (swim_starts<(frame_swim_tcourse.shape[1]-t_post)))]

    vis_starts = np.where((rawdata['stimParam4'][1:]==3) & (rawdata['stimParam4'][:-1]<3))[0] + 1
    vis_ends = np.where((rawdata['stimParam4'][1:]<3) & (rawdata['stimParam4'][:-1]==3))[0] + 1
    vis_starts = frame_tcourse[vis_starts].astype('int')
    vis_ends = frame_tcourse[vis_ends].astype('int')

    swim_starts = np.r_[swim_starts.ravel(), vis_starts.ravel()]
    swim_ends = np.r_[swim_ends.ravel(), vis_ends.ravel()]

    r_swim=np.empty((len(swim_starts), t_len))
    r_swim[:] = 0
    l_swim=np.empty((len(swim_starts), t_len))
    l_swim[:] = 0
    visu=np.empty((len(swim_starts), t_len))
    visu[:] = 0
    swim_len_list = np.zeros(len(swim_starts))

    for i in range(len(swim_starts)):
        swim_len_list[i] = min(swim_ends[i] - swim_starts[i], t_post)
        r_swim[i,:]=frame_swim_tcourse[2,(swim_starts[i]-t_pre):(swim_starts[i]+t_post)]*100000
        l_swim[i,:]=frame_swim_tcourse[1,(swim_starts[i]-t_pre):(swim_starts[i]+t_post)]*100000
        visu[i,:]=-frame_stimParams[0,(swim_starts[i]-t_pre):(swim_starts[i]+t_post)]*10000

    # remove no power swim bout
    remove_ind1 = (r_swim.sum(axis=-1)==0) & (l_swim.sum(axis=-1)==0) & (task_tcourse[swim_starts]<3)
    remove_ind2 = (((r_swim.sum(axis=-1)>0) | (l_swim.sum(axis=-1)>0)) & (task_tcourse[swim_starts]==3))
    remove_ind = remove_ind1 | remove_ind2
    r_swim = r_swim[~remove_ind, :]
    l_swim = l_swim[~remove_ind, :]
    visu = visu[~remove_ind, :]
    swim_starts = swim_starts[~remove_ind]
    swim_ends = swim_ends[~remove_ind]
    
    swim_task_index = np.zeros(len(swim_starts))
    for n in range(len(swim_starts)):
        swim_task_index[n] =  np.median(task_tcourse[swim_starts[n]:swim_ends[n]])
    swim_len_list = swim_len_list[~remove_ind]

    # remove no swim fish
    if len(swim_starts)==0:
        return False

    np.savez(f'swim_power/{folder}_{fish}_swim_dat', \
            swim_starts=swim_starts, swim_ends=swim_ends, \
            r_swim = r_swim, l_swim=l_swim, visu=visu, \
            swim_task_index=swim_task_index)

    print('save swim file')
    return True


if __name__ == "__main__":
    valid_swim_list = []
    for index, row in dat_xls_file.iterrows():
        valid_swim_list.append(valid_swim(row))
    
    swim_xls_file = dat_xls_file[valid_swim_list]
    swim_xls_file.to_csv('depreciated/analysis_sections_ablation_sovo.csv')