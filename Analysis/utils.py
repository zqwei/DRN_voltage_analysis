import numpy as np
import matplotlib.pyplot as plt

# def smooth(a, kernel):
#     b=np.convolve(a,kernel,'same')/np.convolve(np.ones(a.shape),kernel,'same')
#     return b

def smooth(a, kernel):
    return np.convolve(a, kernel, 'full')[kernel.shape[0]//2-1:-kernel.shape[0]//2]


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
        

def baseline(data, window=100, percentile=15, downsample=1, axis=-1):
    """
    Get the baseline of a numpy array using a windowed percentile filter with optional downsampling
    data : Numpy array
        Data from which baseline is calculated
    window : int
        Window size for baseline estimation. If downsampling is used, window shrinks proportionally
    percentile : int
        Percentile of data used as baseline
    downsample : int
        Rate of downsampling used before estimating baseline. Defaults to 1 (no downsampling).
    axis : int
        For ndarrays, this specifies the axis to estimate baseline along. Default is -1.
    """
    from scipy.ndimage.filters import percentile_filter
    from scipy.interpolate import interp1d
    from numpy import ones

    size = ones(data.ndim, dtype='int')
    size[axis] *= window//downsample

    if downsample == 1:
        bl = percentile_filter(data, percentile=percentile, size=size)
    else:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, None, downsample)
        data_ds = data[slices]
        baseline_ds = percentile_filter(data_ds, percentile=percentile, size=size)
        interper = interp1d(range(0, data.shape[axis], downsample), baseline_ds, axis=axis, fill_value='extrapolate')
        bl = interper(range(data.shape[axis]))
    return bl