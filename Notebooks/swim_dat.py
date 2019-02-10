import numpy as np
from pathlib import Path
import pandas as pd
from sys import platform
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt
sns.set_style('ticks')

vol_file = Path('../Voltron_data/Voltron_Log_DRN_Exp.csv')
dat_xls_file = pd.read_csv(vol_file, index_col=0)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
# using Path to handle switches filesystems
if platform == "linux" or platform == "linux2":
    dir_folder = Path('/nrs/ahrens/Ziqiang/Takashi_DRN_project/ProcessedData/')
elif platform == 'win32':
    dir_folder = Path('U:\\Takashi') # put folder for windows system


def valid_swim(row, sig_thres=0.5, ismean=True, isplot=True):
    from scipy.stats import ranksums

    folder = row['folder']
    fish = row['fish']
    task_type = row['task'][0]
    dat_dir = dir_folder/f'{folder}/{fish}/Data/'
    swim_dir = dir_folder/f'{folder}/{fish}/swim/'
    if not os.path.exists(swim_dir/'frame_stimParams.npy'):
        return False
    frame_stimParams = np.load(swim_dir/'frame_stimParams.npy')
    frame_swim_tcourse = np.load(swim_dir/'frame_swim_tcourse_series.npy')
    rawdata = np.load(swim_dir/"rawdata.npy")[()]
    swimdata = np.load(swim_dir/"swimdata.npy")[()]
    reclen=len(swimdata['fltCh1'])
    frame_tcourse=np.zeros((reclen,))
    frame=np.where(np.diff((rawdata['ch3']>3).astype('int'))==1)[0]
    for t in range(len(frame)-1):
        frame_tcourse[frame[t]:frame[t+1]]=t
    swim_starts = frame_tcourse[np.where(swimdata['swimStartT']>0)[0]].astype('int')
    swim_ends = frame_tcourse[np.where(swimdata['swimEndT']>0)[0]].astype('int')
    # swim_starts = np.where(np.diff((frame_swim_tcourse[0,:]>0).astype('int'))==1)[0]
    # swim_ends   = np.where(np.diff((frame_swim_tcourse[0,:]>0).astype('int'))==-1)[0]
    swim_ends   = swim_ends[((swim_starts>50) & (swim_starts<(frame_swim_tcourse.shape[1]-250)))]
    swim_starts = swim_starts[((swim_starts>50) & (swim_starts<(frame_swim_tcourse.shape[1]-250)))]
    task_period = frame_stimParams[2,swim_starts]
    task_index   = frame_stimParams[2,:]+(frame_stimParams[3,:]-1)*4+(frame_stimParams[4,:]-1)*12;
    swim_task_index =  task_index[swim_starts]
    swim_count  = np.zeros((len(swim_starts),))
    if task_type == 'S':
        task_period[:] = 1

    if len(swim_starts)==0:
        return False
    if (task_period==1).sum()<5:
        return False
    if (task_period==2).sum()==0 and not task_type=='S':
        return False

    ind_old=0
    for s in range(len(swim_starts)):
        ind=swim_task_index[s]
        if (ind>ind_old):
            swim_count[s]=1
            ind_old=ind
        elif (ind==ind_old):
            swim_count[s]=swim_count[s-1]+1

    ## mean swim ptterns

    ave_swim=np.empty((len(swim_starts),300))
    ave_swim[:] = 0 #np.nan
    ave_visu=np.empty((len(swim_starts),300))
    ave_visu[:] = 0 #np.nan
    ave_swim_len = np.zeros(len(swim_starts))

    for i in range(len(swim_starts)):
        swim_len = swim_ends[i] - swim_starts[i]
        if swim_len>250:
            swim_len = 250
        ave_swim_len[i] = swim_len
        ave_swim[i,:swim_len+50]=frame_swim_tcourse[2,(swim_starts[i]-50):(swim_starts[i]+swim_len)]*100000
        ave_visu[i,:swim_len+50]=-frame_stimParams[0,(swim_starts[i]-50):(swim_starts[i]+swim_len)]*10000

    gain_stat_len = 100
    gain_stat = np.zeros(gain_stat_len)
    gain_sig_stat = np.ones(gain_stat_len)
    if (task_period==2).sum()>0:
        for ntime in range(gain_stat_len):
            val, pval= ranksums(ave_swim[task_period==1, 50+ntime], ave_swim[task_period==2, 50+ntime])
            gain_stat[ntime] = np.sign(val) * pval
            gain_sig_stat[ntime] = (val>0) and (pval<0.05)
    mean_ = gain_sig_stat.mean()

    if gain_sig_stat.mean()<sig_thres:
        return False

    print(f'{folder} {fish}: average swim difference significance: {mean_}')

    if not isplot:
        return True

    fig, ax = plt.subplots(1, 5, figsize=(16, 4))
    if ismean:
        ax[0].plot(np.arange(-50,250)/300,np.nanmean(ave_swim[task_period==1,:], axis=0), '-k')
    else:
        ax[0].plot(np.arange(-50,250)/300,ave_swim[task_period==1,:].T, '-k')
    if (task_period==2).sum()>0:
        if ismean:
            ax[0].plot(np.arange(-50,250)/300,np.nanmean(ave_swim[task_period==2,:], axis=0), '-r')
        else:
            ax[0].plot(np.arange(-50,250)/300,ave_swim[task_period==1,:].T, '-r')
    ax[0].set_title('Swim power')

    if ismean:
        ax[1].plot(np.arange(-50,250)/300,np.nanmean(ave_visu[task_period==1,:], axis=0), '-k')
    else:
        ax[1].plot(np.arange(-50,250)/300,ave_visu[task_period==1,:].T, '-k')
    if (task_period==2).sum()>0:
        if ismean:
            ax[1].plot(np.arange(-50,250)/300,np.nanmean(ave_visu[task_period==2,:], axis=0), '-r')
        else:
            ax[1].plot(np.arange(-50,250)/300,ave_visu[task_period==1,:].T, '-r')


    ax[1].set_title('Visual velocity')
    tot_swim_power = ave_swim[:,50:].sum(axis=1)
    ax[2].plot(ave_swim_len, tot_swim_power, 'o')
    ax[2].set_ylabel('Total swim power')
    ax[2].set_xlabel('Swim length')
    ax[2].set_title(row['task'])
    ax[3].imshow(ave_swim[:,50:200], aspect='auto')
    ax[3].set_ylabel('Swim index')
    ax[3].set_xlabel('Frame from swim bout onset')

    ax[4].imshow(task_period[:, np.newaxis], aspect='auto', cmap=plt.cm.Set1)
    plt.show()
    plt.savefig(f'swim_power/{folder}_{fish}.png')
    # plt.close()
    return True


def valid_swim_mean(row, ismean=True, isplot=True):
    from scipy.stats import ranksums

    folder = row['folder']
    fish = row['fish']
    task_type = row['task'][0]
    dat_dir = dir_folder/f'{folder}/{fish}/Data/'
    swim_dir = dir_folder/f'{folder}/{fish}/swim/'
    if not os.path.exists(swim_dir/'frame_stimParams.npy'):
        return False
    frame_stimParams = np.load(swim_dir/'frame_stimParams.npy')
    frame_swim_tcourse = np.load(swim_dir/'frame_swim_tcourse_series.npy')
    swim_starts = np.where(np.diff((frame_swim_tcourse[0,:]>0).astype('int'))==1)[0]
    swim_ends   = np.where(np.diff((frame_swim_tcourse[0,:]>0).astype('int'))==-1)[0]
    swim_ends   = swim_ends[((swim_starts>50) & (swim_starts<(frame_swim_tcourse.shape[1]-250)))]
    swim_starts = swim_starts[((swim_starts>50) & (swim_starts<(frame_swim_tcourse.shape[1]-250)))]
    task_period = frame_stimParams[2,swim_starts]
    task_index   = frame_stimParams[2,:]+(frame_stimParams[3,:]-1)*4+(frame_stimParams[4,:]-1)*12;
    swim_task_index =  task_index[swim_starts]
    swim_count  = np.zeros((len(swim_starts),))
    if task_type == 'S':
        task_period[:] = 1

    if len(swim_starts)==0:
        return False
    if (task_period==1).sum()<5:
        return False
    if (task_period==2).sum()==0 and not task_type=='S':
        return False

    ind_old=0
    for s in range(len(swim_starts)):
        ind=swim_task_index[s]
        if (ind>ind_old):
            swim_count[s]=1
            ind_old=ind
        elif (ind==ind_old):
            swim_count[s]=swim_count[s-1]+1

    ## mean swim ptterns

    ave_swim=np.empty((len(swim_starts),300))
    ave_swim[:] = 0 #np.nan
    ave_visu=np.empty((len(swim_starts),300))
    ave_visu[:] = 0 #np.nan
    ave_swim_len = np.zeros(len(swim_starts))
    ave_swim_mean = np.zeros(len(swim_starts))

    for i in range(len(swim_starts)):
        swim_len = swim_ends[i] - swim_starts[i]
        if swim_len>250:
            swim_len = 250
        ave_swim_len[i] = swim_len
        ave_swim[i,:swim_len+50]=frame_swim_tcourse[2,(swim_starts[i]-50):(swim_starts[i]+swim_len)]*100000
        ave_visu[i,:swim_len+50]=-frame_stimParams[0,(swim_starts[i]-50):(swim_starts[i]+swim_len)]*10000
        ave_swim_mean[i] = frame_swim_tcourse[2,(swim_starts[i]):(swim_starts[i]+swim_len)].mean()*100000

    gain_sig_stat = True
    if (task_period==2).sum()>0:
        val, pval= ranksums(ave_swim_mean[task_period==1], ave_swim_mean[task_period==2])
        gain_sig_stat = (val>0) and (pval<0.05)

    if not gain_sig_stat:
        return False

    print(f'{folder} {fish}: average swim difference is significant')

    if not isplot:
        return True

    fig, ax = plt.subplots(1, 5, figsize=(16, 4))
    if ismean:
        ax[0].plot(np.arange(-50,250)/300,np.nanmean(ave_swim[task_period==1,:], axis=0), '-k')
    else:
        ax[0].plot(np.arange(-50,250)/300,ave_swim[task_period==1,:].T, '-k')
    if (task_period==2).sum()>0:
        if ismean:
            ax[0].plot(np.arange(-50,250)/300,np.nanmean(ave_swim[task_period==2,:], axis=0), '-r')
        else:
            ax[0].plot(np.arange(-50,250)/300,ave_swim[task_period==1,:].T, '-r')
    ax[0].set_title('Swim power')

    if ismean:
        ax[1].plot(np.arange(-50,250)/300,np.nanmean(ave_visu[task_period==1,:], axis=0), '-k')
    else:
        ax[1].plot(np.arange(-50,250)/300,ave_visu[task_period==1,:].T, '-k')
    if (task_period==2).sum()>0:
        if ismean:
            ax[1].plot(np.arange(-50,250)/300,np.nanmean(ave_visu[task_period==2,:], axis=0), '-r')
        else:
            ax[1].plot(np.arange(-50,250)/300,ave_visu[task_period==1,:].T, '-r')


    ax[1].set_title('Visual velocity')
    tot_swim_power = ave_swim[:,50:].sum(axis=1)
    ax[2].plot(ave_swim_len, tot_swim_power, 'o')
    ax[2].set_ylabel('Total swim power')
    ax[2].set_xlabel('Swim length')
    ax[2].set_title(row['task'])
    ax[3].imshow(ave_swim[:,50:200], aspect='auto')
    ax[3].set_ylabel('Swim index')
    ax[3].set_xlabel('Frame from swim bout onset')

    ax[4].imshow(task_period[:, np.newaxis], aspect='auto', cmap=plt.cm.Set1)
    plt.show()
    plt.savefig(f'swim_power_mean/{folder}_{fish}.png')
    # plt.close()
    return True

