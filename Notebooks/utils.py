import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt
from scipy.stats import sem, ranksums, ttest_ind, spearmanr
from scipy.stats import ttest_1samp, trim_mean, wilcoxon
from scipy.stats import f_oneway
sns.set(font_scale=2, style='ticks')

# def smooth(a, kernel):
#     b=np.convolve(a,kernel,'same')/np.convolve(np.ones(a.shape),kernel,'same')
#     return b

def smooth(a, kernel):
    return np.convolve(a, kernel, 'full')[kernel.shape[0]//2:-(kernel.shape[0]//2)]


def test_smooth(kernel):
    x = np.zeros(1001)
    x[30] = 1
    y = smooth(x, kernel)
    assert np.argmax(y)==30 # this is not working for boxcar kernel


def boxcarKernel(sigma=60):
    kernel = np.ones(sigma)
    return kernel/kernel.sum()


def gaussKernel(sigma=20):
    kernel = (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(np.arange(-sigma*3,sigma*3+1)**2)/(2*sigma**2))
    return kernel/kernel.sum()


def bootstrap_p_ABtest(test, ctrl, is_left=True):
    import bootstrapped.bootstrap as bs
    import bootstrapped.stats_functions as bs_stats
    import bootstrapped.compare_functions as bs_comp
    num_ = 1000 # use for Multi-test correction
    p_val = 0.05
    boot_strp_iter = max(int(num_/p_val)*10+10, 10000)
    num_thread = 30
    test_ctrl_dist = bs.bootstrap_ab(test, ctrl, bs_stats.mean, bs_comp.difference, return_distribution=True, num_threads=num_thread, num_iterations=boot_strp_iter)
    if is_left:
        return (test_ctrl_dist<0).mean()
    else:
        return (test_ctrl_dist>0).mean()


def cont_mode(data, isplot=False):
    from scipy import stats
    kde = stats.gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 100)
    p = kde(x)
    if isplot:
        plt.plot(x, p)
    return x[np.argmax(p)]


def plt_raster(spk_list, c='k', f_=300, t_shift=100, mz=10):
    for n, ntrial in enumerate(spk_list):
        t_ = np.where(ntrial==1)[0]-t_shift
        plt.plot(t_/f_, np.ones(len(t_))*n, f'.{c}', markersize=mz)
        

def shaded_errorbar(x, y, error, ax=plt, color='k'):
    ax.plot(x, y, '-', lw=1, color=color)
    ax.fill_between(x, y-error, y+error, facecolor=color, lw=0, alpha=0.8)


def search_paired_data(row, flist):
    if 'before' not in row['fish']:
        return False
    fish = row['fish'][:-6]
    for _, row_ in flist.iterrows():
        if row_['folder'] != row['folder']:
            continue
        if row_['fish'] == fish+'after':
            return True
    return False


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


def spk_shape(spk_list, dff):
    spk_ = []
    for t_ in spk_list:
        if t_+30<len(dff):
            spk_.append(dff[t_-30:t_+30])
    spk_ = np.array(spk_)
    return spk_.mean(axis=0)


def search_paired_data(row, flist):
    if 'before' not in row['fish']:
        return False
    fish = row['fish'][:-6]
    for _, row_ in flist.iterrows():
        if row_['folder'] != row['folder']:
            continue
        if row_['fish'] == fish+'after':
            return True
    return False