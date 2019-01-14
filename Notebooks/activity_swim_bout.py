import numpy as np
from pathlib import Path
import pandas as pd
from sys import platform
import os
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.stats import sem

# using Path to handle switches filesystems
if platform == "linux" or platform == "linux2":
    dir_folder = Path('/nrs/ahrens/Ziqiang/Takashi_DRN_project/ProcessedData/')
elif platform == 'win32':
    dir_folder = Path('U:\\Takashi') # put folder for windows system

def mean_spk_sub(row, isplot=False):
    folder = row['folder']
    fish = row['fish']
    print(f'Processing {folder} {fish}')
    dat_dir = dir_folder/f'{folder}/{fish}/Data/'
    swim_dir = dir_folder/f'{folder}/{fish}/swim/'
    dff = np.load(dat_dir/'Voltr_spikes.npz')['voltrs']
    dff = dff - np.median(dff, axis=1, keepdims=True)
    spk = np.load(dat_dir/'Voltr_spikes.npz')['spk']
    num_cell = spk.shape[0]
    spk = np.r_['-1', np.zeros((num_cell, 600)), spk]
    frame_stimParams = np.load(swim_dir/'frame_stimParams.npy')
    frame_swim_tcourse = np.load(swim_dir/'frame_swim_tcourse.npy')
    
    subvolt = dff.copy()
    for n, ndff in enumerate(dff):
        subvolt[n, :] = medfilt(ndff, kernel_size=51)
        
    swim_starts = np.where(np.diff((frame_swim_tcourse[0,:]>0).astype('int'))==1)[0]
    swim_ends   = np.where(np.diff((frame_swim_tcourse[0,:]>0).astype('int'))==-1)[0]
    swim_ends   = swim_ends[((swim_starts>50) & (swim_starts<(frame_swim_tcourse.shape[1]-250)))]
    swim_starts = swim_starts[((swim_starts>50) & (swim_starts<(frame_swim_tcourse.shape[1]-250)))]
    task_period = frame_stimParams[2,swim_starts]
    task_index  = frame_stimParams[2,:]+(frame_stimParams[3,:]-1)*4+(frame_stimParams[4,:]-1)*12;
    swim_task_index = task_index[swim_starts]
    swim_count  = np.zeros((len(swim_starts),))

    ind_old=0
    for s in range(len(swim_starts)):
        ind=swim_task_index[s]
        if (ind>ind_old):
            swim_count[s]=1
            ind_old=ind
        elif (ind==ind_old):
            swim_count[s]=swim_count[s-1]+1

    n_spk_bin = 6
    stimParam = frame_stimParams[2,:]+(frame_stimParams[3,:]-1)*4
    ntrials = int(max(frame_stimParams[4,:])-1)
    t_spk = -300
    t_bin = 50
    
    spk_bout_list = np.zeros((num_cell, 2, 3))
    sub_bout_list = np.zeros((num_cell, 2, 3))

    for c in range(num_cell):
        ave_resp_spk = np.zeros((len(swim_starts),n_spk_bin));
        ave_resp_dff = np.zeros((len(swim_starts),300));
        for i in range(len(swim_starts)):
            for n_ in range(n_spk_bin):
                ss_ = swim_starts[i]+t_spk+n_*t_bin
                se_ = swim_starts[i]+t_spk+n_*t_bin + t_bin
                if se_ <0:
                    continue
                if swim_starts[i]+t_spk+n_*t_bin<0:
                    ss_ = 0
                ave_resp_spk[i,n_]  = spk[c,ss_:se_].mean()*300
            sub_ = subvolt[c,(swim_starts[i]-50):(swim_starts[i]+250)]
            ave_resp_dff[i,:len(sub_)] = sub_
        ave_resp_dff -= ave_resp_dff[:,:30].mean(axis=1)[:,None]

        peak_resp = ave_resp_dff.min(axis=-1)
        cum_resp = ave_resp_dff.sum(axis=-1)
        peak_resp_time = np.argmin(ave_resp_dff, axis=-1)
        ave_resp_spk[np.isnan(ave_resp_spk)] = 0
        mean_spk = ave_resp_spk[:, :].mean(axis=-1)
        val_to_plot = cum_resp
        
        for n_period in range(1, 3):
            spk_bout_list[c, n_period-1, 0] = mean_spk[((task_period==n_period) & (swim_count<=3))].mean()
            sub_bout_list[c, n_period-1, 0] = val_to_plot[((task_period==n_period) & (swim_count<=3))].mean()
            spk_bout_list[c, n_period-1, 1] = mean_spk[((task_period==n_period) & (swim_count<=6) & (swim_count>=3))].mean()
            sub_bout_list[c, n_period-1, 1] = val_to_plot[((task_period==n_period) & (swim_count<=6) & (swim_count>=3))].mean()
            spk_bout_list[c, n_period-1, 2] = mean_spk[((task_period==n_period) & (swim_count<=9) & (swim_count>=6))].mean()
            sub_bout_list[c, n_period-1, 2] = val_to_plot[((task_period==n_period) & (swim_count<=9) & (swim_count>=6))].mean()
        
        
        if isplot:
            fig, ax = plt.subplots(1, 6, figsize=(28, 3))
            # for n_period in range(1, 3):
            #     ax[n_period-1].plot(mean_spk[((task_period==n_period) & (swim_count<=3))], val_to_plot[((task_period==n_period) & (swim_count<=3))],'ob')
            #     ax[n_period-1].plot(mean_spk[((task_period==n_period) & (swim_count<=6) & (swim_count>=3))], val_to_plot[((task_period==n_period) & (swim_count<=6) & (swim_count>=3))],'og')
            #     ax[n_period-1].plot(mean_spk[((task_period==n_period) & (swim_count<=9) & (swim_count>=6))], val_to_plot[((task_period==n_period) & (swim_count<=9) & (swim_count>=6))],'or')
            #     ax[n_period-1].set_xlabel('average spike rate')
            #     ax[n_period-1].set_ylabel('total inhibition')
            #     ax[n_period-1].set_title(f'task epoch {n_period}')

            for n_period in range(1, 3):
                x = mean_spk[((task_period==n_period) & (swim_count<=3))].mean()
                xrr = sem(mean_spk[((task_period==n_period) & (swim_count<=3))])
                y = val_to_plot[((task_period==n_period) & (swim_count<=3))].mean()
                yrr = sem(val_to_plot[((task_period==n_period) & (swim_count<=3))])
                ax[n_period-1].errorbar(x, y, xerr=xrr, yerr=yrr, fmt='ob')

                x = mean_spk[((task_period==n_period) & (swim_count<=6) & (swim_count>=3))].mean()
                xrr = sem(mean_spk[((task_period==n_period) & (swim_count<=6) & (swim_count>=3))])
                y = val_to_plot[((task_period==n_period) & (swim_count<=6) & (swim_count>=3))].mean()
                yrr = sem(val_to_plot[((task_period==n_period) & (swim_count<=6) & (swim_count>=3))])
                ax[n_period-1].errorbar(x, y, xerr=xrr, yerr=yrr, fmt='og')

                x = mean_spk[((task_period==n_period) & (swim_count<=9) & (swim_count>=6))].mean()
                xrr = sem(mean_spk[((task_period==n_period) & (swim_count<=9) & (swim_count>=6))])
                y = val_to_plot[((task_period==n_period) & (swim_count<=9) & (swim_count>=6))].mean()
                yrr = sem(val_to_plot[((task_period==n_period) & (swim_count<=9) & (swim_count>=6))])
                ax[n_period-1].errorbar(x, y, xerr=xrr, yerr=yrr, fmt='or')
                ax[n_period-1].set_xlabel('average spike rate')
                ax[n_period-1].set_ylabel('total inhibition')
                ax[n_period-1].set_title(f'task epoch {n_period}')

            box_list = []
            for n_period in range(1, 3):
                box_list.append(val_to_plot[((task_period==n_period) & (swim_count<=3))])
            ax[2].boxplot(box_list, positions=list(range(1,3)))
            ax[2].set_title('Swim bout #0 - #2', color='b')
            ax[2].set_ylabel('total inhibition')
            ax[2].set_xlabel('task epoch')

            box_list = []
            for n_period in range(1, 3):
                box_list.append(val_to_plot[((task_period==n_period) & (swim_count<=6) & (swim_count>=3))])
            ax[3].boxplot(box_list, positions=list(range(1,3)))
            ax[3].set_title('Swim bout #3 - #5', color='g')
            ax[3].set_ylabel('total inhibition')
            ax[3].set_xlabel('task epoch')

            box_list = []
            for n_period in range(1, 3):
                box_list.append(val_to_plot[((task_period==n_period) & (swim_count<=9) & (swim_count>=6))])
            ax[4].boxplot(box_list, positions=list(range(1,3)))
            ax[4].set_title('Swim bout #6 - #8', color='r')
            ax[4].set_ylabel('total inhibition')
            ax[4].set_xlabel('task epoch')


            box_list = []
            for n_period in range(1, 3):
                box_list.append(mean_spk[task_period==n_period])
            ax[5].boxplot(box_list, positions=list(range(1,3)))
            ax[5].set_title('Average spike rate')
            ax[5].set_ylabel('Spike before epoch')
            ax[5].set_xlabel('task epoch')

            plt.show()
    
    return spk_bout_list, sub_bout_list
        
    
    