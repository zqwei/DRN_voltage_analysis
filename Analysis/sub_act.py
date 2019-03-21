from scipy.stats import mannwhitneyu as utest
from scipy.stats import ranksums
from scipy.stats import ttest_ind
import numpy as np
from matplotlib import pyplot as plt

def sub_process(dff, k_size=51):
    from scipy.signal import medfilt
    subvolt = dff.copy()
    for n, ndff in enumerate(dff):
        subvolt[n, :] = medfilt(ndff, kernel_size=k_size)
    return subvolt

def sub_input(ave_, c='k', t_min = 0, t_max = 400, t_zero=100, isplot=False):
    mean_ = ave_[:, t_min:t_max].mean(axis=0)
    mean_ = mean_ - mean_[:(100-t_min)].max()
    max_ = mean_.max()
    min_ = mean_.min()
    std_ = ave_[:, t_min:t_max].std(axis=0)/np.sqrt(ave_.shape[0])
    act_ = mean_[t_zero-60:t_zero+270].reshape(-1, 30)

    if isplot:
        plt.plot(np.arange(t_max-t_min)/300-(t_zero-t_min)/300, mean_, f'-{c}', lw=2)
        plt.plot(np.arange(t_max-t_min)/300-(t_zero-t_min)/300, mean_-std_, f'--{c}', lw=0.5)
        plt.plot(np.arange(t_max-t_min)/300-(t_zero-t_min)/300, mean_+std_, f'--{c}', lw=0.5)
        plt.plot(np.arange(-60, 270, 30)/300+15/300, act_.mean(axis=-1), f'o{c}', ms=15, alpha=0.7)

    
    act_pre_ = act_[0]
    dist = act_.max()-act_.min()
    exc_input = []
    t_vec = np.arange(-60, 270, 30)/300+15/300
    pre_input = 0
    for nperiod in range(act_.shape[0]-1):
        _, p = ttest_ind(act_pre_, act_[nperiod+1])
        diff_ = (act_[nperiod+1].mean()-act_pre_.mean())/dist
        if p > .05 or np.abs(diff_)<0.1:
            act_pre_ = np.concatenate([act_pre_, act_[nperiod+1]])
            exc_input.append(pre_input)
        else:
            act_pre_ = act_[nperiod+1]
            pre_input = np.sign(diff_)
            exc_input.append(pre_input)
            

    exc_input = np.array(exc_input)
    exc_input[1] = 0
    act_mean = act_.mean(axis=-1)
    
    # print(exc_input.shape)
    # print(act_mean)
    
    ### inh
    # on time
    try:
        inh_on_ind = np.where(exc_input==-1)[0][0]+1
        inh_on_t = t_vec[inh_on_ind-1] + np.argmax(act_[inh_on_ind-1])/300
    except:
        inh_on_t = np.nan
    # peak time
    try:
        inh_peak_ind = np.where((exc_input[1:]-exc_input[:-1])==2)[0][0]+1
    except:
        if exc_input[-1]==-1:
            inh_peak_ind = -1
    try:
        inh_peak_t = t_vec[inh_peak_ind] + np.argmin(act_[inh_peak_ind])/300
    except:
        inh_peak_t = np.nan
    try:
        inh_ = act_mean[inh_peak_ind] - act_mean[inh_on_ind-1]
        if inh_ > 0:
            inh_ = act_mean[inh_peak_ind] - act_mean[:inh_peak_ind].max()
    except:
        inh_ = 0
        
    ### exc
    # on time
    try:
        exc_on_ind = np.where(exc_input==1)[0][0]+1
        exc_on_t = t_vec[exc_on_ind-1] + np.argmin(act_[exc_on_ind-1])/300
    except:
        exc_on_t = np.nan
    try:
        exc_peak_ind = np.where((exc_input[1:]-exc_input[:-1])==-2)[0][0]+1
    except:
        if exc_input[-1]==1:
            exc_peak_ind = -1
    # peak time
    try:
        exc_peak_t = t_vec[exc_peak_ind] + np.argmax(act_[exc_peak_ind])/300
    except:
        exc_peak_t = np.nan
    try:
        exc_ = act_mean[exc_peak_ind] - act_mean[exc_on_ind-1]
        if exc_ < 0:
            exc_ = act_mean[exc_peak_ind] - act_mean[:exc_peak_ind].min()
    except:
        exc_ = 0

    return [inh_on_t, inh_peak_t, inh_], [exc_on_t, exc_peak_t, exc_],
