import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *
from scipy.stats import sem, ranksums

k_spk = gaussKernel(sigma=1)
k_sub = gaussKernel(sigma=1)
t_pre = 100 # time window pre-swim
t_post = 350 # time window post-swim
swim_power_thres = 100
t_swim_CL = t_pre + 100
t_swim_OL = t_pre + 200
t_label = np.arange(-t_pre, t_post)/300
c_list = ['k', 'r', 'b']
labels = ['CL', 'Swim-only', 'Visual-only']

def sovo_act(folder, fish, is_plot=False):
    if not os.path.exists(f'../Analysis/swim_voltr/{folder}_{fish}_swim_voltr_dat.npz'):
        return None    
    
    _ = np.load(f'../Analysis/swim_power/{folder}_{fish}_swim_dat.npz')
    task_period = _['swim_task_index'].astype('int')            
    _ = np.load(f'../Analysis/swim_voltr/{folder}_{fish}_swim_voltr_dat.npz')
    sub_swim = _['sub_swim']
    spk_swim = _['spk_swim']
    trial_valid = _['trial_valid']
    
    _ = np.load(f'../Analysis/swim_power/{folder}_{fish}_swim_dat.npz')
    r_swim = _['r_swim']
    l_swim = _['l_swim']
    task_period = _['swim_task_index'].astype('int')
    visu = _['visu']
    p_swim = np.sqrt(r_swim**2 + l_swim**2)
    
#     trial_pre = (p_swim[:, :t_pre]>0).sum(axis=-1)==0
#     trial_valid_CL = (p_swim[:, t_swim_CL:t_swim_CL+150]>0).sum(axis=-1)==0
#     trial_valid_CL = trial_valid_CL & trial_pre
#     trial_valid_OL = ((visu.max(axis=-1, keepdims=True)-visu)[:, :-50]>0).sum(axis=-1)==0
#     trial_valid_OL = trial_valid_OL & trial_pre & ((p_swim[:, t_swim_OL:t_pre+300]>1).sum(axis=-1)==0)
#     # trial_valid_OL = trial_valid_OL & ((p_swim[:, t_swim_OL:t_swim_OL+150]>0).sum(axis=-1)==0)
#     trial_valid_VL = (p_swim[:, t_pre:t_pre+300]>0).sum(axis=-1)==0
#     if np.percentile(p_swim[(task_period==1) & trial_valid].mean(axis=0), 95)>swim_power_thres:
#         return None
#     if np.percentile(p_swim[(task_period==2) & trial_valid].mean(axis=0), 95)>swim_power_thres:
#         return None
#     if ((task_period==2) & trial_valid & trial_valid_OL).sum()<5:
#         return None

    trial_pre = (p_swim[:, :t_pre]>0).sum(axis=-1)==0
    trial_valid_CL = (p_swim[:, t_swim_CL:t_swim_CL+150]>0).sum(axis=-1)==0
    trial_valid_CL = trial_valid_CL & trial_pre
    trial_valid_OL = ((visu.max(axis=-1, keepdims=True)-visu)[:, :-50]>0).sum(axis=-1)==0
    # trial_valid_OL = trial_valid_OL & (p_swim[:, t_swim_CL:t_pre+300].max(axis=-1)<swim_power_thres)
    trial_valid_OL = trial_valid_OL & trial_pre & ((p_swim[:, t_swim_OL:t_pre+300]>1).sum(axis=-1)==0)
    trial_valid_OL = trial_valid_OL & trial_pre
    # trial_valid_OL = trial_valid_OL & ((p_swim[:, t_swim_OL:t_pre+300]>1).sum(axis=-1)==0)
    # trial_valid_OL = trial_valid_OL & ((p_swim[:, t_swim_OL:t_swim_OL+150]>0).sum(axis=-1)==0)
    trial_valid_VL = (p_swim[:, t_pre:t_pre+300]>0).sum(axis=-1)==0
    # trial_valid_VL = trial_valid_VL & (visu[:, t_swim_OL:t_pre+300].min(axis=-1)>=0)
    # trial_valid_VL = trial_valid_VL & trial_pre
    
    if np.percentile(p_swim[(task_period==1) & trial_valid].mean(axis=0), 95)>swim_power_thres:
        return None
    
    if np.percentile(p_swim[(task_period==2) & trial_valid].mean(axis=0), 95)>swim_power_thres:
        return None
        
    if ((task_period==2) & trial_valid & trial_valid_OL).sum()<5:
        return None
    
    sub_ave = []
    spk_ave = []
    fish_id = []
    
    for n_cell in range(sub_swim.shape[0]):
        sub_list = sub_swim[n_cell]
        tmp = []
        for n_spk in sub_list:
            tmp.append(smooth(n_spk, k_sub))
        sub_list = np.array(tmp)
        sub_list = sub_list - sub_list[:, (t_pre-60):t_pre].mean(axis=-1, keepdims=True)
        spk_list = spk_swim[n_cell]
        tmp = []
        for n_spk in spk_list:
            tmp.append(smooth(n_spk, k_spk))
        spk_list = np.array(tmp)
        fish_id.append(folder+fish[:5])
        
        if is_plot:
            fig, ax = plt.subplots(1, 2, figsize=(10, 3))
            ax = ax.flatten()
            
        tmp1 = []
        tmp2 = []
        for n in range(3):
            if n==0:
                trial_valid_ = trial_valid & trial_valid_CL
            if n==1:
                trial_valid_ = trial_valid & trial_valid_OL  #& trial_valid_CL
            if n==2:
                trial_valid_ = trial_valid & trial_valid_VL
            ave_ = sub_list[(task_period==n+1) & trial_valid_, :]*100
            mean_ = np.mean(ave_, axis=0)
            if is_plot:
                std_ = sem(ave_, axis=0, nan_policy='omit')
                ax[0].plot(t_label, mean_, f'-{c_list[n]}', lw=2)
                ax[0].plot(t_label, mean_-std_, f'--{c_list[n]}', lw=0.5)
                ax[0].plot(t_label, mean_+std_, f'--{c_list[n]}', lw=0.5)
                ax[0].set_xlim([-t_pre/300, t_post/300])
                ax[0].set_xlabel('Time (sec)')
                ax[0].set_ylabel('dF/F')
                sns.despine()
            tmp1.append(mean_)
            
            ave_ = spk_list[(task_period==n+1) & trial_valid_, :]*300
            mean_ = np.mean(ave_, axis=0)
            if is_plot:
                std_ = sem(ave_, axis=0, nan_policy='omit')
                ax[1].plot(t_label, mean_, f'-{c_list[n]}', lw=2, label=labels[n])
                ax[1].plot(t_label, mean_-std_, f'--{c_list[n]}', lw=0.5)
                ax[1].plot(t_label, mean_+std_, f'--{c_list[n]}', lw=0.5)
                ax[1].set_xlim([-t_pre/300, t_post/300])
                ax[1].set_xlabel('Time (sec)')
                ax[1].set_ylabel('Spikes')
                sns.despine()
            tmp2.append(mean_)
        sub_ave.append(np.array(tmp1))
        spk_ave.append(np.array(tmp2))
        
        if is_plot:
            plt.show()
    return sub_ave, spk_ave, fish_id