"""
This file supplies the functions to identify valid swims in swim-only-visual-only task.

Valid swim is defined as the fish adapts to the gain

Created on 05/08/2019
@author: Ziqiang Wei
@email: weiz@janelia.hhmi.org
"""

import numpy as np
import pandas as pd
import os

vol_file = '../Voltron_data/Voltron_Log_DRN_Exp.csv'
dat_xls_file = pd.read_csv(vol_file, index_col=0)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
dir_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/ProcessedData/'
t_pre = 100 # time window pre-swim
t_post = 350 # time window post-swim
t_len = t_pre+t_post


def valid_swim(row):
    folder = row['folder']
    fish = row['fish']
    task_type = row['task'] # task type names
    swim_dir = dir_folder + f'{folder}/{fish}/swim/'

    frame_stimParams = np.load(swim_dir+'frame_stimParams.npy')
    frame_swim_tcourse = np.load(swim_dir+'frame_swim_tcourse_series.npy')
    rawdata = np.load(swim_dir+"rawdata.npy", allow_pickle=True)[()]
    swimdata = np.load(swim_dir+"swimdata.npy", allow_pickle=True)[()]
    reclen=len(swimdata['fltCh1'])
    frame_tcourse=np.zeros((reclen,))
    frame=np.where(np.diff((rawdata['ch3']>3).astype('int'))==1)[0]
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
    
    swim_task_index =  task_tcourse[swim_starts]
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
        if 'Swimonly' in row['task']: 
            valid_swim_list.append(valid_swim(row))
        else:
            valid_swim_list.append(False)
    
    swim_xls_file = dat_xls_file[valid_swim_list]
    swim_xls_file.to_csv('depreciated/analysis_sections_based_on_swim_pattern_sovo.csv')