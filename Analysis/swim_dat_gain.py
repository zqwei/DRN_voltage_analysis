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

# vol_file = '../Voltron_data/depreciated/Voltron_Log_DRN_Exp_update.csv'
vol_file = '../Voltron_data/Voltron_Log_DRN_Exp.csv'
dat_xls_file = pd.read_csv(vol_file, index_col=0)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
dir_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/ProcessedData/'
t_pre = 100 # time window pre-swim
t_post = 350 # time window post-swim
t_len = t_pre+t_post
gain_stat_len = 100 # time length to examine the gain adaption after swim


def valid_swim(row, sig_thres=0.5):
    from scipy.stats import ranksums
    folder = row['folder']
    fish = row['fish']
    task_type = row['task'] # task type names
    
    # only analysis gain adaption and memory task
    if (not ('Gain adaptation' in task_type)) and (not ('Raphe memory task' in task_type)):
        return False
    # remove after-ablation data
    if 'after ablation' in task_type:
        return False
    
    swim_dir = dir_folder + f'{folder}/{fish}/swim/'
    if not os.path.exists(swim_dir+'frame_stimParams.npy'):
        return False
    
    print(swim_dir)
    
    frame_stimParams = np.load(swim_dir+'frame_stimParams.npy')
    frame_swim_tcourse = np.load(swim_dir+'frame_swim_tcourse_series.npy')
    rawdata = np.load(swim_dir+"rawdata.npy", allow_pickle=True)[()]
    swimdata = np.load(swim_dir+"swimdata.npy", allow_pickle=True)[()]
    reclen=len(swimdata['fltCh1'])
    frame_tcourse=np.zeros((reclen,))
    frame=np.where(np.diff((rawdata['ch3']>3).astype('int'))==1)[0]
    for t in range(len(frame)-1):
        frame_tcourse[frame[t]:frame[t+1]]=t
    swim_starts = frame_tcourse[np.where(swimdata['swimStartT']>0)[0]].astype('int')
    swim_ends = frame_tcourse[np.where(swimdata['swimEndT']>0)[0]].astype('int')
    # collect trial within t-pre, and t-post valid range
    swim_ends   = swim_ends[((swim_starts>t_pre) & (swim_starts<(frame_swim_tcourse.shape[1]-t_post)))]
    swim_starts = swim_starts[((swim_starts>t_pre) & (swim_starts<(frame_swim_tcourse.shape[1]-t_post)))]

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
    remove_ind = (r_swim.sum(axis=-1)==0) & (l_swim.sum(axis=-1)==0)
    r_swim = r_swim[~remove_ind, :]
    l_swim = l_swim[~remove_ind, :]
    visu = visu[~remove_ind, :]
    swim_starts = swim_starts[~remove_ind]
    swim_ends = swim_ends[~remove_ind]

    task_period = frame_stimParams[2,swim_starts]
    task_index   = frame_stimParams[2,:]+(frame_stimParams[3,:]-1)*4+(frame_stimParams[4,:]-1)*12
    swim_task_index =  task_index[swim_starts]
    swim_len_list = swim_len_list[~remove_ind]
    swim_count  = np.zeros((len(swim_starts),))

    ind_old=0
    for s in range(len(swim_starts)):
        ind=swim_task_index[s]
        if (ind>ind_old):
            swim_count[s]=1
            ind_old=ind
        elif (ind==ind_old):
            swim_count[s]=swim_count[s-1]+1

    # remove no swim fish
    if len(swim_starts)==0:
        return False
    if (task_period==1).sum()<5:
        return False

    ## mean swim ptterns
    # compare the swim power for the first x frames after swim onset
    gain_stat = np.zeros(gain_stat_len)
    gain_sig_stat = np.ones(gain_stat_len)
    if (task_period==2).sum()>0:
        for ntime in range(gain_stat_len):
            val, pval= ranksums(r_swim[task_period==1, t_pre+ntime], r_swim[task_period==2, t_pre+ntime])
            gain_stat[ntime] = np.sign(val) * pval
            gain_sig_stat[ntime] = (val>0) and (pval<0.05)

    if (gain_sig_stat.mean()<sig_thres):
        return False

    print(f'{folder} {fish}: average swim difference significance: {gain_sig_stat.mean()}')

    np.savez(f'swim_power/{folder}_{fish}_swim_dat', \
            swim_starts=swim_starts, swim_ends=swim_ends, \
            r_swim = r_swim, l_swim=l_swim, visu=visu, \
            task_period = task_period, swim_task_index=swim_task_index)

    print('save swim file')
    return True


if __name__ == "__main__":

    valid_swim_list = []
    for index, row in dat_xls_file.iterrows():
        valid_swim_list.append(valid_swim(row, sig_thres=0.5))
    
    swim_xls_file = dat_xls_file[valid_swim_list]
    swim_xls_file.to_csv('depreciated/analysis_sections_gain.csv')